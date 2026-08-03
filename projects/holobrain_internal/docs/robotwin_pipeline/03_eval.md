# 03 — RoboTwin 2.0 评测

## 1. 两条路线，只有集群这条跑通

### 1.1 本机路线 —— 被 curobo 卡死，**不要再试**

脚本 `common/scripts/eval_robotwin_ckpt11.sh` 是为本机写的，做了完整的
"备份 ckpt → 组装 EVAL_MODEL_DIR → staging policy → 起评测"，
甚至有 conda env 冒烟测试和 GPU 显存 preflight。**但它跑不完**：

```
File ".../robotwin/envs/robot/robot.py", line 15
    from .planner import CuroboPlanner
ImportError: cannot import name 'CuroboPlanner' from 'envs.robot.planner'
```

根因：`envs/robot/robot.py:15` **无条件** `from .planner import CuroboPlanner`，
而 `planner.py` 里 `class CuroboPlanner` 定义在 `try: import curobo ... except:` 块内 ——
curobo 装不上 ⇒ 类不存在 ⇒ ImportError。且 `robot.py` 里大量 `CuroboPlanner(...)` 和
`isinstance(..., CuroboPlanner)`，**curobo 是硬依赖**，不是可选加速。

curobo 在本机装不上（GitHub timeout；hobot mirror 上 `curobo` / `Nvidia-curobo` 都是
<1 KB 占位包；PyPI 上是 squatter）。完整排查记录见
[`../claude_tasks/2026-07-22_robotwin_eval_env_ready_blocked_curobo.md`](../claude_tasks/2026-07-22_robotwin_eval_env_ready_blocked_curobo.md)。

失败的 driver log 留在
`/jfs-public/users/kun01.wu/robo_orchard_lab/workspace/checkpoints_backup/eval_ckpt11_20260722_093622.log`
（末尾是 `ZeroDivisionError` —— 因为 `results` 为空还去求平均，见 §5.3）。

**融合 conda env `robotwin_holobrain_eval` 是装好了的**（155 包，46/46 冒烟通过，
模型能 `to('cuda')` 用 2.7 GB），只差 curobo。所以本机路线只要拿到 curobo 就能通，
但既然集群这条已经跑通，没必要再折腾。

### 1.2 集群路线 —— 走通了

curobo 在 docker 镜像里，所以直接提交 AIDI job 即可：

```bash
cd /home/users/kun01.wu-labs/git_repo/robo_orchard_lab
RoboOrchardJob-AIDISubmit submit_from_config \
    --config projects/holobrain_internal/common/aidi_submit_config/submit_cfg_robotwin_eval_kun_mydocker.json
```

镜像
`docker.hobot.cc/imagesys/kun01.wu/holobrain-eval:ubuntu22.04-gcc11.4-py3.10-cuda12.8-torch280-robotwin-20260724-v3`
（自建）。**v1 / v2 都在起容器阶段就挂了**（Failed，没留下 main.log，只有空的 `shell_logs`），
**只有 v3 跑通**，复现请直接用 v3 这个 tag。

07-23 那次用的是上游公共镜像
`robotlab-mani:ubuntu2204-gcc11.4-cu128-nccl2277-torch280-erdma`（配置
`submit_cfg_robotwin_eval_kun.json`），也跑通了。两个镜像结果一致（详见 [07_results.md](07_results.md)）。

---

## 2. 提交配置

`aidi_submit_config/submit_cfg_robotwin_eval_kun_mydocker.json` 关键字段：

```jsonc
{
  "num_workers": 1, "gpu_per_worker": 8, "wall_time": 14400,   // 1 pod × 8 卡 × 4h
  "input_bucket": "robot_lab,robot_lab2",
  "workspace_folder": "/jfs-public/users/kun01.wu/robo_orchard_lab/aidi_workspace/...",
  "cmd": [
    "export PYTHONPATH=${WORKING_PATH}:$PYTHONPATH",
    "export HYDRA_FULL_ERROR=1",
    "ln -s /horizon-bucket/robot_lab2/users/tianwei.lin/data/robotwin2/assets assets",
    "python3 robotwin_eval.py --task_config demo_clean --task_names <16 个> ",
    "  --model_config '/horizon-bucket/.../ckpts/checkpoint_11_eval' ",
    "  --vlm_ckpt_dir /horizon-bucket/robot_lab/users/xuewu.lin/ckpt ",
    "  --urdf_dir /horizon-bucket/robot_lab2/datasets/all_data/urdf/urdf_v20260711 ",
    "  --model_processor robotwin2_0_processor --model_prefix model --test_num 50"
  ],
  "to_upload": [
    "robo_orchard_lab",
    "projects/holobrain_internal/common/holobrain_utils.py",
    "projects/holobrain_internal/common/holobrain_robotwin_policy",
    "projects/holobrain_internal/common/robotwin_eval.py",
    "robotwin/envs", "robotwin/description", "robotwin/task_config", "robotwin/script"
  ]
}
```

三点值得注意：

- **`assets` 靠 `ln -s` 引外部 bucket 目录**（tianwei.lin 的），不在 `to_upload` 里 —— 太大。
- **`robotwin/` 四个子目录要提交**：`envs`（任务定义）、`description`（instruction）、
  `task_config`（demo_clean.yml）、`script`（eval_policy.py）。少一个就跑不起来。
  提交前仓库根需要有 `robotwin` 这个 symlink 指向本地副本。
- **`workspace_folder` 是 JFS 绝对路径**，不是仓库内相对路径 —— 否则每次提交会在仓库根
  生成一份代码快照污染 `git status`。**不能放 bucket**（`clear_workspace` 要 `rmtree`）。

与上游模板 `submit_cfg_robotwin_eval.json` 的差别只有三处：`job_name`、`workspace_folder`、
`--model_config`。16 任务列表、`--test_num 50`、`--task_config demo_clean` 全部沿用上游。

---

## 3. 调用链与并行方式

```
robotwin_eval.py  (1 个主进程)
  ├─ :126-132  按 index % num_gpus 把 16 个 task 分配到 8 张卡 → 每卡 2 个
  ├─ :134-148  每张卡 fork 一个 multiprocessing.Process(eval_tasks, gpu_id, [task…])
  │             结果通过 multiprocessing.Manager().dict() 汇总
  └─ 每个子进程 eval_tasks(): 对分到的 task **串行**执行
        :42-43   env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        :44-70   subprocess.run(["python3", "script/eval_policy.py",
                                 "--config", "holobrain_robotwin_policy/deploy_policy.yml",
                                 "--overrides", ...])
                 stdout+stderr → <log_dir>/log.txt
        :71-85   从 log.txt 末尾往回扫 "Success rate"，正则抓百分比写进 results
```

`script/eval_policy.py` 内部：

```
:167  st_seed = 100000 * (1 + seed)          # seed 来自 deploy_policy.yml，本次 = 0
:212  succ_seed = 0 ; :219 now_seed = st_seed
:225  while succ_seed < test_num:            # test_num = 50
        用 expert TASK_ENV.play_once() 试当前 seed
        不合法 → :242/:251 now_seed += 1，跳过，不计数
        合法   → policy 跑一次，:257 succ_seed += 1, :260 now_seed += 1
:347  print(f"Success rate: {suc}/{test_num} => {pct}%, current seed: {now_seed}")
```

**HoloBrain 不是 client-server**：`holobrain_robotwin_policy/deploy_policy.py` 里
`HoloBrainPolicy` 直接 `ModelMixin.load_model(...)` 然后 `self.model(data)` ——
模型与 sapien env 在**同一个 Python 进程、同一张卡**。这也是本机路线需要"融合 env"的原因。

`deploy_policy.py:156-157`：每次 `get_action` 出 64 步，**只取前 32 步执行**
（`valid_action_step = 32`）。

---

## 4. 产物结构 —— 集群与本机不一样

`script/eval_policy.py:126-128`：

```python
if <本机>:
    save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}")
else:                                    # 集群（CLUSTER 环境变量存在）
    save_dir = Path(f"/job_data/{task_name}/{task_config}")
```

**集群下是扁平的 `/job_data/<task>/<task_config>/`，没有 policy/ckpt/timestamp 那几层。**
（`../eval_robotwin_ckpt11.md` §5.2 描述的是本机的嵌套结构，集群不适用。）

AIDI 把 `/job_data` 归档成 `output/`，所以最终：

```
log/<job-id>-task-0-main.log            robotwin_eval.py 主 stdout，末尾是 16 任务汇总 JSON
output/<task_name>/demo_clean/
    log.txt                             eval_policy.py 全量 stdout（逐 episode）
    _result.txt                         成功率小数（:185,:190 写）
    episode0.mp4 … episode49.mp4        每 episode 一段（demo_clean.yml:eval_video_log=true）
```

汇总 JSON（`robotwin_eval.py:152-157`）：

```json
{ "<task>": 46.0, ..., "num_tasks": 16, "mean": 43.875, "test_num_per_task": 50 }
```

**这份 JSON 只在 stdout 里，不落文件** —— 要拿它必须去读 `main.log`。

归档后的结果见 [07_results.md](07_results.md) §5。

---

## 5. 三个坑

### 5.1 失败的 task 被静默丢出统计

`robotwin_eval.py:71-89`：只有 `returncode == 0` 才写 `results`；非零时只打一行
`Fail to eval task[...] with returncode N`，**该 task 不进 results**，
而 `mean = sum(results.values()) / len(results)`（`:153`）照算。

→ **看 `mean` 之前先看 `num_tasks`。** 小于 16 就是幸存者偏差，不能跨 run 比较。

### 5.2 `--test_num` 与 `demo_clean.yml:episode_num` 是两回事

命令行 `--test_num 50` 才是实际 trial 数；`demo_clean.yml` 里的 `episode_num: 100`
是数据采集时的配置，评测走的是前者。

### 5.3 全部 task 失败时报的是 `ZeroDivisionError`

`robotwin_eval.py:153` 在 `results` 为空时直接除以 `len(results)=0`。
所以**看到 `ZeroDivisionError: division by zero` 不要去查数学问题，
它的真实含义是"16 个 task 一个都没跑成"** —— 去看
`output/<task>/demo_clean/log.txt` 里的真实报错（本机那次就是 curobo 的 ImportError）。

---

## 6. 换 checkpoint 重跑

只改两处：

```jsonc
"job_name": "eval_robotwin_holobrain_<新名字>",
// cmd 里：
"  --model_config '/horizon-bucket/robot_lab/users/kun01.wu/robo_orchard_lab/ckpts/<新 ckpt 目录>' ",
```

`<新 ckpt 目录>` 必须是**组装好的 deploy package**，不是 accelerate state。四件套 + 两项：

```
model.safetensors                       从 accelerate ckpt 拷
model.config.json                       同上
robotwin2_0_processor.json              从 workspace/ 顶层拷（train.py:96-110 导出）
robotwin2_0_inference.config.json       同上
urdf/                                   从 workspace/urdf/ 拷
ckpt -> /horizon-bucket/robot_lab/users/xuewu.lin/ckpt    symlink，VLM base weight
```

最后那个 symlink 不能少 —— `model.config.json` 里 `vlm_pretrain='./ckpt/Qwen2.5-VL-3B-Instruct'`
是**相对 model_config 目录**的路径。组装步骤可参考
`common/scripts/eval_robotwin_ckpt11.sh` 的 `[1/5]` `[2/5]` 两段（这部分与 curobo 无关，是好的）。

一次约 3 小时 / 8 卡。按 [07_results.md](07_results.md) §4.1，
**要做 checkpoint 之间的比较，50 trial 不够，必须跑两遍或加大 `--test_num`。**
