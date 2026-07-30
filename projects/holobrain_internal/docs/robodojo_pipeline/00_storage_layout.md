# 00 · 存储落点约定（权威）

产出该放哪，以本文为准。2026-07-30 起生效。

## 一句话

> **代码留 `/home`；会变的训练状态放 JFS；定版产物放 bucket 的项目根。**

## 三个落点

| 落点 | 路径 | 放什么 | 关键语义 |
|---|---|---|---|
**LOCAL** | `/home/users/kun01.wu-labs/git_repo/robo_orchard_lab/` | 只放代码、配置、文档、`.git` | POSIX 完整；`/home` 28 T |
**JFS** | `/jfs-public/users/kun01.wu/robo_orchard_lab/` | 训练输出、所有缓存、`TMPDIR`、wandb | **POSIX 完整**（rename/delete 都行）；947 T |
**BUCKET** | `/horizon-bucket/robot_lab/users/kun01.wu/robo_orchard_lab/` | 定版 ckpt、评测结果归档 | FUSE 层拒 rename/delete/rmdir/sqlite；**但 `aidi-inf-cli bkt-file rm/mv` 可以删改** |

注意：JFS 与 BUCKET 上的用户目录名是 **`kun01.wu`**，不是 unix 用户名 `kun01.wu-labs`。

## 目录结构

```
LOCAL  git_repo/robo_orchard_lab/
         projects/holobrain_internal/common/
           workspace -> JFS/workspace          ← symlink，训练输出穿过它落 JFS
           ckpt      -> bucket/xuewu.lin/ckpt   ← 输入：VLM 底座
           data      -> robot_lab2/.../all_data ← 输入：数据集
           urdf      -> robot_lab2/.../urdf_*   ← 输入：机器人模型
         robo_orchard_lab_env.sh                ← 用前 source

JFS    robo_orchard_lab/
         workspace/checkpoints/checkpoint_N/    ← accelerate 断点（含 optimizer.bin，可 resume）
         workspace/checkpoints_backup/          ← 手动冷备，躲开 rolling 删除
         cache/{hf,torch,pip,uv,xdg,triton,matplotlib,wandb}/
         tmp/ wandb/ share/ config/
         manifests/  tools/                     ← 校验清单与工具

BUCKET robo_orchard_lab/
         ckpts/<run_name>/checkpoint_<step>/    ← 定版 deploy package（评测 job 读这里）
         eval_results/<run_name>/               ← 评测结果归档
         README.md
```

## 判据是访问模式，不是文件类别

**任何会被 rename / delete / sqlite 打开的路径，一律不进 BUCKET。**

- `workspace/checkpoints/` **绝对不能**放 BUCKET：accelerate 以 rolling `total_limit=3` 写它，
  **要 delete 最旧的 checkpoint**，而 BUCKET 在 FUSE 层拒绝 delete。
- `HF_HOME` 等缓存**绝对不能**放 BUCKET：HuggingFace 靠 `*.incomplete` → rename 落盘。
- 2026-07-30 用 18 项独立探测实测：BUCKET 上 `flock`、`mmap` 写、`hardlink`、`chmod`
  **其实都可用**，被拒的只有 rename / delete / rmdir / sqlite。
  所以「旁边有 `.lock` 文件」不是排除 BUCKET 的理由。

## 什么是自动的，什么要手动

| 场景 | 落点 | 自动？ |
|---|---|---|
本地 `train.py`（CWD=`common/`，`--workspace` 默认 `./workspace`） | JFS | ✅ 穿 symlink，自动 |
本地缓存 / `TMPDIR` | JFS | ⚠️ **必须先 `source robo_orchard_lab_env.sh`**，否则回落 `$HOME` 和 97% 满的 `/` |
**AIDI 训练 job**（`--workspace /job_data`） | job PFS | ❌ **不自动**，见下 |
**AIDI 评测 job**（`--eval_result_dir` 默认 `/job_data/robodojo_eval_results`） | job PFS | ❌ **不自动** |

### AIDI 的产出不会自己进 bucket

**这一点以前的文档写错了。** `/job_data` 是 job 自己的 PFS 输出挂载，
`aidictl job logs list/cat` 能看到（显示为 `output/`）。**平台不会把它同步到
`users/<user>/aidi_output/`** —— `users/kun01.wu-labs/` 这个路径根本不存在，
bucket 里那些 `aidi_output/*` 目录是以前的 session **手工建并手工 cp** 的。

所以 job 跑完必须自己归档：

```bash
# 1) 把 job 产出取到本地（大文件别用 aidictl download，会静默截断）
#    小文件用 aidictl job logs cat；ckpt 走 PFS HTTP + curl -C -
# 2) 用统一脚本归档到 bucket 项目根，自动校验 md5
projects/holobrain_internal/scripts/archive_to_bucket.sh ckpts        <run_name> <本地目录>
projects/holobrain_internal/scripts/archive_to_bucket.sh eval_results <run_name> <本地目录>
```

`archive_to_bucket.sh` 会：拒绝落到项目根之外 → `cp -rd`（保留 symlink，不跟随）
→ 逐文件 md5 双边校验 → 写清单到 `JFS/manifests/`。

## AIDI 提交暂存目录也已移出仓库

每次 `submit_from_config` 都会把 `to_upload` 里的代码 `rsync -aL` 到 `workspace_folder`。
默认值是**相对路径**，于是快照落进仓库根 —— 这就是那 8 个
`submit-holobrain-*/` 目录的来源（约 50 M，还会污染 `git status`）。

本项目的 13 个 submit cfg 已改成绝对路径：

```
/jfs-public/users/kun01.wu/robo_orchard_lab/aidi_workspace/<原名>
```

**为什么能这么改、为什么只能放 JFS**：`workspace_folder` 在
`robo_orchard_jobs/job_submit/submit_config.py:72` 是裸 `str`，
`aidi/job_config.py:264-281` 只对它做 `os.path.exists` / `shutil.rmtree` /
`os.makedirs` / `os.path.join` / `rsync -aL`，**没有强制拼仓库根**，所以绝对路径生效。
但 `clear_workspace` 要 `rmtree`、上传要 `rsync -aL`（临时文件 + rename），
**两者都被 bucket 拒绝**，所以只能放 JFS。

这些动作都发生在**提交时的开发机上**，所以路径写错会在几秒内失败，不会浪费一个 6 小时的 job。

## 命名

- `ckpts/<run_name>/checkpoint_<step>/` —— `run_name` 用 job_name，`step` 用**训练步数**
  （如 `checkpoint_100000`），不要用 accelerate 的保存序号（`checkpoint_20`），两者容易混。
- **bucket 上名字一次写对**：FUSE 层改不了名；虽然 `aidi-inf-cli bkt-file mv` 能改，
  但那是控制面操作，不如一次写对。

## 自查

```bash
source /home/users/kun01.wu-labs/git_repo/robo_orchard_lab/robo_orchard_lab_env.sh
rol_env_check      # 17 个环境变量落点 + workspace symlink + 断链扫描
rol_verify_ckpt    # 真的打开 checkpoint_20，比对张量数（基线 1087）
```

## 两个已知坑

1. **`.gitignore` 尾斜杠不匹配 symlink**：`workspace/` 换成 symlink 后
   `.gitignore:191` 的 `…/workspace/` 失效，`git status` 会冒出 untracked。
   修法是写 `.git/info/exclude`（本地私有），**不要改共享的 `.gitignore`**。
2. **`rsync --sparse` 在 FUSE 上慢 19 倍**（22 vs 420 MB/s）。本仓库没有稀疏文件，
   往 JFS/bucket 拷一律**别加 `--sparse`**，加 `--inplace` 避免临时文件残留。
