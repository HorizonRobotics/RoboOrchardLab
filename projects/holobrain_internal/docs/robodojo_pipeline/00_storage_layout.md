# 00 · 产物落点约定（权威）

**本项目跑训练/评测时，每一类产物具体落在哪。** 2026-07-30 起生效，经逐项审计。

## 一句话

> **`/home` 只放代码；会变的训练状态放 JFS；定版产物放 BUCKET 的项目根。**

| 层 | 路径 | 容量 | 关键语义 |
|---|---|---|---|
**LOCAL** | `/home/users/kun01.wu-labs/git_repo/robo_orchard_lab/` | 28 T（9 T 空闲） | POSIX 完整 |
**LOCAL** | `/`（含默认 `/tmp`） | 446 G，**常年 98% 满** | 别往这写 |
**JFS** | `/jfs-public/users/kun01.wu/robo_orchard_lab/` | 940 T 空闲 | **POSIX 完整**（18 项探测全过） |
**BUCKET** | `/horizon-bucket/robot_lab/users/kun01.wu/robo_orchard_lab/` | 544 P | 可写**不可删**（详见 §6） |

> **JFS/BUCKET 上的用户目录名是 `kun01.wu`，不是登录名 `kun01.wu-labs`。** 极易写错。

---

## 1. 总表：哪类产物落哪

| 产物 | 落点 | 由什么决定 | 自动？ |
|---|---|---|---|
**accelerate 训练断点**（含 `optimizer.bin`，可续训） | `JFS/workspace/checkpoints/checkpoint_N/` | `--workspace` 默认 `./workspace` + symlink | ✅ |
训练日志 / tensorboard | `JFS/workspace/logs/` | `train.py:224` 从 `--workspace` 派生 | ✅ |
config 快照 | `JFS/workspace/configs/` | `train.py:66` | ✅ |
`*_processor.json`、`*_inference.config.json` | `JFS/workspace/` | `train.py:98,106` | ✅ |
手动冷备的 ckpt | `JFS/workspace/checkpoints_backup/` | 人工 `cp` | 手动 |
**所有缓存**（HF / torch / pip / uv / triton / matplotlib / XDG / wandb） | `JFS/cache/*`、`JFS/wandb/` | `robo_orchard_lab_env.sh` | ⚠️ **要先 source** |
**临时文件** | `JFS/tmp/` | `TMPDIR`（同上） | ⚠️ 同上 |
AIDI 提交的代码快照 | `JFS/aidi_workspace/<name>/` | submit cfg 的 `workspace_folder` | ✅ |
逐文件 md5 清单 | `JFS/manifests/` | 归档脚本 | ✅ |
迁移/校验工具 | `JFS/tools/` | — | — |
**定版 deploy package**（只有权重，用于评测） | `BUCKET/robo_orchard_lab/ckpts/<run>/checkpoint_<step>/` | `archive_to_bucket.sh` | ❌ **手动归档** |
**评测结果归档** | `BUCKET/robo_orchard_lab/eval_results/<run>/` | `archive_to_bucket.sh` | ❌ **手动归档** |
汇总后的小 JSON（进仓库） | `docs/robodojo_pipeline/results/{20k,100k}/` | 人工 commit | 手动 |
**CUDA JIT / PTX 缓存** | `~/.nv` | `CUDA_CACHE_PATH` | ✅ **刻意留 LOCAL** |
**torch C++/CUDA 扩展（`.so`）** | `~/.cache/torch_extensions` | `TORCH_EXTENSIONS_DIR` | ✅ **刻意留 LOCAL** |
Omniverse / Isaac shader cache | `~/.cache/ov`、`~/.local/share/ov`、`~/.nvidia-omniverse` | 不认 XDG，天然留本地 | ✅ 刻意 |
AIDI job 运行期产物 | job 自己的 PFS（`/job_data`、`/job_tboard`） | submit cfg | ✅ 但**跑完必须归档** |

**为什么编译产物必须留 LOCAL**：`.so` 放网络盘会拖慢甚至破坏动态链接。
`TORCH_EXTENSIONS_DIR` 显式钉住，就是防止哪天有人改了 `XDG_CACHE_HOME` 把 `.so` 带到 JFS 上。

---

## 2. 本地训练：一步步落在哪

```bash
cd /home/users/kun01.wu-labs/git_repo/robo_orchard_lab
source ./robo_orchard_lab_env.sh          # ← 不做这一步，缓存和 TMPDIR 会回落 $HOME 和 98% 满的 /
cd projects/holobrain_internal/common
python train.py --config configs/config_holobrain_common.py ...
```

`--workspace` 默认 `./workspace`，而 `common/workspace` 是一条**指向 JFS 的绝对路径 symlink**，
所以以下全部自动落 JFS，代码一行不用改：

```
common/workspace -> /jfs-public/users/kun01.wu/robo_orchard_lab/workspace
  checkpoints/checkpoint_N/     accelerate 断点（rolling total_limit=3，会删最旧的）
  checkpoints_backup/           想留住某个 step 就手动 cp 到这里，躲开 rolling 删除
  logs/                         --logging_dir 默认从 workspace 派生
  configs/                      本次运行的 config 快照
  *_processor.json / *_inference.config.json
```

**`--workspace` 这个默认值出现在 3 个入口，全都一致**（审计确认）：
`common/train.py:217`、`common/export.py:121`、`common/data_visualize/video.py:89`。

### ⚠️ 训练输出为什么绝不能放 BUCKET

accelerate 用 rolling `total_limit=3` 写 checkpoints，**要 delete 最旧的那一份**，
而 BUCKET 在 POSIX 层拒绝 delete。放上去训练会中途失败。

---

## 3. AIDI 集群训练：产出不会自动进 bucket

```bash
RoboOrchardJob-AIDISubmit submit_from_config \
    --config projects/holobrain_internal/common/aidi_submit_config/submit_cfg_robodojo_train_100k.json
```

| 阶段 | 落点 |
|---|---|
提交时（**在开发机上**）把 `to_upload` 的代码 `rsync -aL` 过去 | `JFS/aidi_workspace/<workspace_folder 叶子名>/` |
job 运行时 `train.py --workspace /job_data` | **job 自己的 PFS**（`aidictl job logs list` 里显示为 `output/`） |
`--logging_dir /job_tboard` | 同上 |
跑完 | **不会自动进 bucket，必须手动归档 ↓** |

### ‼️ 没有「AIDI 自动 rsync 到 bucket」这回事

`users/kun01.wu-labs/` 这种"自动落点"**根本不存在**（早期文档写错过）。
bucket 里那些 `aidi_output/*` 目录全是人手建、人手 `cp` 的。
**job 产出只留在 job PFS 里，会随 job 过期。**

### 正确的归档方式

```bash
# 1) 把 job 产出取到本地。大文件别用 aidictl job logs download —— 会静默截断且照样 exit 0。
#    小文件用 aidictl job logs cat；ckpt 走 PFS HTTP + curl -C - 续传。
# 2) 用统一脚本归档，落点由脚本强制、并做逐文件 md5 双边校验
projects/holobrain_internal/scripts/archive_to_bucket.sh ckpts \
    holobrain_robodojo_posttrain_v10/checkpoint_50000  ~/tmp/ckpt50k
```

脚本会：拒绝越出项目根 → `cp -rd`（**保留 symlink，绝不跟随**，否则会把同事那一大坨拷进来）
→ 逐文件 md5 校验 → 清单写 `JFS/manifests/` → 目标已存在且内容不同时**拒绝覆盖**。

---

## 4. AIDI 集群评测：结果同样要自己取回来

```bash
RoboOrchardJob-AIDISubmit submit_from_config \
    --config .../submit_cfg_robodojo_eval_kun_20k.json
```

| 参数 | 默认值 | 落在哪 |
|---|---|---|
`--model_dir` | 在 cfg 里显式给 | 读 `BUCKET/robo_orchard_lab/ckpts/<run>/checkpoint_<step>/` |
`--eval_result_dir` | `/job_data/robodojo_eval_results`（`robodojo_eval.py:183`） | job PFS |
`--kit_cache` | `/job_data/.cache/isaacsim-kit`（:188） | job PFS |
`--runtime_env_dir` | `/tmp/robodojo-env-config`（:177） | pod 内的 `/tmp`，job 本地 |

取回与归档：

```bash
# 一次拿到所有任务的 SR（比逐个解析 _result.json 快得多；注意路径是 log/ 不是 log/run_0/）
aidictl job logs cat <job_id> "log/<job_id>-task-0-main.log" | grep 'finished: success_rate'

# 合并两批 job 并产出官方口径 summary
cd projects/holobrain_internal/scripts
python aggregate_robodojo_results.py --gen-job <25ep_job> --nongen-job <50ep_job> \
    --label 20k --out-dir <某个 JFS 或本地目录>

# 归档到 bucket
./archive_to_bucket.sh eval_results robodojo-<run_name> <上面的 out-dir>
```

小的 summary JSON 可以直接 commit 进 `docs/robodojo_pipeline/results/`（几百 KB，方便 diff 和审计）。

### ⚠️ 本地直接跑 `robodojo_eval.py` 的一个坑

`--runtime_env_dir` 默认写死 `/tmp/robodojo-env-config`，**不认 `TMPDIR`**。
在集群 pod 里无妨（job 本地），但**在开发机上直接跑会落到 98% 满的 `/`**。
本地跑时显式给：`--runtime_env_dir "$TMPDIR/robodojo-env-config"`。

---

## 5. 只读输入（不要往里写）

| 路径 | 内容 |
|---|---|
`common/ckpt -> /horizon-bucket/robot_lab/users/xuewu.lin/ckpt` | 同事的 Qwen VLM 底座权重 |
`common/data -> /horizon-bucket/robot_lab2/datasets/all_data` | 数据集（含 `robodojo/lmdb/*`） |
`common/urdf -> …/all_data/urdf/urdf_v20260711` | 机器人模型 |
`/horizon-bucket/robot_lab/users/kun01.wu/datasets/RoboDojo/Assets/` | RoboDojo 仿真资产，**跨项目共享** |

前三条 symlink**不是随手建的**：AIDI 训练 cfg 的 `cmd` 里在集群侧做了一模一样的 `ln -s`，
目的是让本地和集群的目录形状完全一致，同一份代码两边都能跑。

---

## 6. BUCKET 的实测语义（务必先读再往上写）

18 项逐项独立探测（2026-07-30）：

| 操作 | 结果 |
|---|---|
`create` `overwrite` `append` `random-write` `truncate` | ✅ |
`flock` `mmap 写` `hardlink` `chmod` `symlink` | ✅ —— 流传的"加锁/mmap 不能用"是**错的** |
`rename` `delete` `rmdir` | ❌ EACCES |
`sqlite` 写 | ❌ disk I/O error（它要 rename/unlink journal） |

**删除：本账号没有权限，连控制面也没有。**
`aidi-inf-cli bkt-file rm` 这个命令存在，但需要桶级的「写任意和命令行删除」权限；
`aidi-inf-cli bkt ls` 实测该列**为空**，真跑报 `bucket permission denied` code 7。
→ **对我们而言 BUCKET 是「能写不能删」。** 要删只有找管理员，或走
`aidi-inf-cli remove-task submit … --approver <人>` 审批流（可 restore）。

**所以名字必须一次写对。** 排除 BUCKET 的判据只有两个：
**要 rename（含一切"原子写"）**，或**将来可能要删**。

冷读吞吐（清页缓存后，同一个 2.84 GB 文件）：
LOCAL 1676 MB/s · JFS 326 MB/s · BUCKET 218 MB/s。
JFS 只快 1.49×，**单次顺序加载权重直接从桶里读没问题**；
要反复读同一份数据（多 epoch dataloader）就先 materialize 到 JFS。

---

## 7. 自查

```bash
source /home/users/kun01.wu-labs/git_repo/robo_orchard_lab/robo_orchard_lab_env.sh
rol_env_check      # 20 个环境变量落点 + workspace symlink + 断链扫描
rol_verify_ckpt    # 真的打开 checkpoint_20，比对张量数（基线 1087）
```

`rol_env_check` 期望输出：**18 个 JFS + 2 个「LOCAL（刻意）」+「需要注意的写入点: 0 个」+ 断链 0**。
任何一项是 `!! 仍在 $HOME` 或 `!! 落在 97% 满的 /`，说明有东西漏了。

---

## 8. 已知遗留 / 需要注意

| 项 | 状态 |
|---|---|
**不 source 环境文件 = 等于没迁** | 缓存和 `TMPDIR` 会回落 `$HOME` 与 `/`。这是最容易漏的一步 |
`robodojo_eval.py` 的 `--runtime_env_dir` 写死 `/tmp` | 本地跑要显式覆盖，见 §4 |
9 个**上游** submit cfg 的 `workspace_folder` 仍是相对路径 | 提交它们会在仓库根留下代码快照。其中 4 个（`submit/`、`submit-eval-holobrain-libero{,-plus}`、`submit-value-model`）不匹配 `.gitignore` 的 `/submit-holobrain*/`，已在 `.git/info/exclude` 本地兜底。**属别人的配置，没有代改** |
`checkpoint_{18,19,20}` 内部各有一份 2.84 G 重复 | `model.safetensors` 与 `model/model.safetensors` 逐字节相同且非 hardlink，共 8.1 G。删前需先确认 accelerate 的 `load_state` 只读一份。在 JFS 上，不删无害 |
bucket 上新旧两份并存（多 10 G） | 归集前没验证删除权限，拷完才发现删不掉。清单见 `ADMIN_DELETE_REQUEST.md`，等管理员执行 |

## 9. 两条容易反复踩的坑

1. **`.gitignore` 的尾斜杠不匹配 symlink。** `…/workspace/` 只匹配目录；
   把它换成 symlink 之后忽略规则失效，`git status` 会冒出 untracked。
   修法是把**无尾斜杠**路径写进 **`.git/info/exclude`**（本地私有），
   **不要改共享的 `.gitignore`** —— 那会把个人决定推给所有同事。
2. **`rsync --sparse` 在 FUSE 上慢 19 倍**（22 vs 420 MB/s）。本仓库没有稀疏文件，
   往 JFS/BUCKET 拷一律**别加 `--sparse`**；加 `--inplace` 可避免临时文件残留。
   怀疑存储慢之前，先 `dd bs=8M conv=fsync` 单独量一次挂载点。
