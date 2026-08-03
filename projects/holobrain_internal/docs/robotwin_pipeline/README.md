# HoloBrain × RoboTwin 2.0 Pipeline — 后训练与评测实录

本目录记录 **HoloBrain 在 RoboTwin 2.0 上的一轮后训练 + 评测**：从本机单卡后训练，
到 checkpoint 组装成部署包，再到 AIDI 集群侧 16 任务评测。

**语境**：kun01.wu 在 2026-07-21 → 07-24 实操的实录，2026-07-31 回溯整理。
与同目录的 [`../robodojo_pipeline/`](../robodojo_pipeline/) 是**两条独立的工作线**——
RoboTwin 这条在前（07-21~07-24），RoboDojo 那条在后（07-25 起）。

> **想直接看结果** → [07_results.md](07_results.md)。
> **想知道当时怎么被卡住的** → [`../claude_tasks/2026-07-22_robotwin_eval_env_ready_blocked_curobo.md`](../claude_tasks/2026-07-22_robotwin_eval_env_ready_blocked_curobo.md)。

---

## ⚠️ 三条读之前必须知道的

1. **所有 RoboTwin 数字都来自 step 60000 的中间 ckpt（`checkpoint_11`），不是终版。**
   训练实际跑满了 step 100000（`checkpoint_19`/`20`），**终版权重从未在 RoboTwin 上评测过**。
   不要把本目录的数字当作"这次后训练的最终成绩"引用。

2. **仓库当前的 `configs/dataset_specs.py` 不是当时训练用的那份。**
   当时 `filter_list` 只留了 `robotwin1_0` + `robotwin2_0`；现在是全量（07-28 为 RoboDojo 改回的）。
   权威快照在 `/jfs-public/users/kun01.wu/robo_orchard_lab/workspace/configs/`，
   由 `train.py:61-68` 每次启动 copytree 生成。详见 [01_training.md](01_training.md)。

3. **这轮后训练的 loss 曲线已经不存在了。** 原因与排查过程见 [01_training.md](01_training.md) §4。

---

## 架构总览

```
┌──────────── 后训练 (本机单卡，非集群 job) ──────────────────┐
│                                                              │
│  cd projects/holobrain_internal/common                      │
│  python3 train.py --config configs/config_holobrain_common.py│
│      GPU 0 独占，PID 1540974，无 accelerate launch           │
│      起点权重 = v9 holobrain_v9_newinit/checkpoint_50 (HTTP) │
│      数据 = 仅 robotwin1_0 + robotwin2_0                     │
│                                                              │
│  → workspace/checkpoints/checkpoint_N/                       │
│     每 5k step 一份 accelerate state，total_limit=3 滚动删除  │
│     (workspace 是指向 JFS 的绝对路径 symlink)                │
│                                                              │
│  2026-07-21 05:19 → 07-22 12:16，跑满 max_step=1e5           │
└──────────────────────────────────────────────────────────────┘
                    ▼  (中途 07-22 01:16 手工冷备 checkpoint_11)
┌──────────── 组装 deploy package (手工) ────────────────────┐
│                                                              │
│  checkpoints_backup/checkpoint_11_step60000/  (accelerate state)│
│      model.safetensors      ─┐                               │
│      model.config.json      ─┤                               │
│  workspace/                  ├→  ckpts/checkpoint_11_eval/   │
│      robotwin2_0_processor.json        (bucket, 2.7 G)       │
│      robotwin2_0_inference.config.json                       │
│      urdf/                  ─┘                               │
│  + ckpt -> xuewu.lin/ckpt  (symlink, VLM base weight)        │
│                                                              │
│  脚本: common/scripts/eval_robotwin_ckpt11.sh 的 [1/5][2/5]  │
└──────────────────────────────────────────────────────────────┘
                    ▼
┌──────────── 评测 (1 pod × 8 GPU) ──────────────────────────┐
│                                                              │
│  submit_cfg_robotwin_eval_kun_mydocker.json                 │
│      │  RoboOrchardJob-AIDISubmit submit_from_config         │
│      ▼                                                       │
│  cluster pod:                                                │
│    python3 robotwin_eval.py --task_config demo_clean         │
│           --task_names <16 个> --test_num 50                 │
│      → 16 任务按 index % 8 分到 8 张卡，每卡串行 2 个任务     │
│      → 每任务 fork: python3 script/eval_policy.py            │
│           --config holobrain_robotwin_policy/deploy_policy.yml│
│         └─ HoloBrainPolicy 直接 load_model + self.model(data)│
│            (非 client-server，模型与 sapien env 同进程同卡)   │
│                                                              │
│  → /job_data/<task>/demo_clean/{log.txt,_result.txt,*.mp4}   │
│     = AIDI 归档的 output/                                     │
│  → 汇总 JSON 打在主 stdout = 归档的 log/<job>-task-0-main.log │
└──────────────────────────────────────────────────────────────┘
```

---

## 文档结构

| 文件 | 内容 |
|---|---|
| [01_training.md](01_training.md) | 后训练侧：启动方式、起点权重、**数据集口径（当时 vs 现在）**、超参、ckpt 落点、**loss 为什么没了** |
| [03_eval.md](03_eval.md) | 评测侧：两条路线（本机卡 curobo / 集群自建镜像）、任务与 seed 口径、并行方式、产物结构、`robotwin_eval.py:71` 静默丢任务的坑 |
| [07_results.md](07_results.md) | **结果**：16 任务 × 50 trial 的两次完整评测对照表、分层解读、三条读数注意事项。配套 JSON 在 `results/{20260723,20260724}/` |

编号刻意与 `../robodojo_pipeline/` 对齐（01 训练 / 03 评测 / 07 结果），便于横向对照；
中间空号是那边有而这边没有的环节。

---

## 关键数字速查

| Quantity | Value | Source |
|---|---|---|
| Benchmark | RoboTwin 2.0, `demo_clean`, embodiment `aloha-agilex` | `task_config/demo_clean.yml` |
| Eval tasks | **16** | `submit_cfg_robotwin_eval_kun_mydocker.json` cmd |
| Trials / task | **50** (`--test_num 50`) | 同上 |
| 评测权重 | `checkpoint_11` = **step 60000** | `custom_checkpoint_0.pkl:global_step_id=59999` |
| 训练终点 | step **100000** (`checkpoint_19`)，**未评测** | `scheduler.bin:last_epoch=100000` |
| 训练数据 | 仅 `robotwin1_0` + `robotwin2_0` | `workspace/configs/dataset_specs.py:613-614` |
| 采样权重 | **未生效**（`use_dataset_sample_weights = False`） | `workspace/configs/dataset_specs.py:707` |
| 起点权重 | v9 `holobrain_v9_newinit/checkpoint_50` | `config_holobrain_common.py:89` |
| Batch | 16（单卡单进程，无梯度累积） | `config_holobrain_common.py:29` |
| Prediction horizon | `pred_steps=64`, `chunk_size=4` → 16 chunks | `config_holobrain_common.py:21-22` |
| VLM | Qwen2.5-VL-3B-Instruct，取前 4 层，**不冻结** | `config_holobrain_common.py:86,42` |
| Decoder | 10 层，`embed_dims=384`，`multi_modal_attn=True` | `config_holobrain_common.py:86-90` |
| 每次 policy 输出用几步 | `valid_action_step=32` | `holobrain_robotwin_policy/deploy_policy.py:156` |
| Seed 起点 | `st_seed = 100000×(1+seed)`，`seed=0` | `deploy_policy.yml:seed`; `script/eval_policy.py:167` |
| Mean SR | **42.625%** (07-23) / **43.875%** (07-24) | 两个 job 主 log 末尾 |

---

## Job 速查

| Job ID | 名称 | 时间 | 镜像 | 结果 |
|---|---|---|---|---|
| `bcloud-bj-zone1-51192e413238` | `eval_robotwin_holobrain_ckpt11_kun` | 07-23 09:53→13:02 | `robotlab-mani:...-erdma`（公共） | Succeeded，mean **42.625%** |
| `bcloud-bj-zone1-a74cf470f80b` | `..._kun_mydocker_v3` | 07-24 13:47→16:57 | `kun01.wu/holobrain-eval:...-v3`（自建） | Succeeded，mean **43.875%** |
| `bcloud-bj-zone1-6642e2b705d6` | `..._kun_mydocker_v2` | 07-24 12:04 | 自建 v2 | Failed，起容器阶段挂，无 main.log |
| `bcloud-bj-zone1-c4493e14d78b` | `..._kun_mydocker` | 07-24 10:52 | 自建 v1 | Failed，同上 |
| 另 4 个 | `..._kun` | 07-22 19:41~19:50 | 公共 | Stopped（调试中主动停） |

---

## 产物落点

| 内容 | 路径 |
|---|---|
| 训练状态（可 resume，含 optimizer.bin） | `/jfs-public/users/kun01.wu/robo_orchard_lab/workspace/checkpoints/checkpoint_{18,19,20}/` |
| checkpoint_11 冷备 | `.../workspace/checkpoints_backup/checkpoint_11_step60000/` |
| 评测用 deploy package | `/horizon-bucket/robot_lab/users/kun01.wu/robo_orchard_lab/ckpts/checkpoint_11_eval/` |
| **评测结果归档** | `/horizon-bucket/robot_lab/users/kun01.wu/robo_orchard_lab/eval_results/robotwin-holobrain-ckpt11-{20260723,20260724}/` |
| 训练数据 | `/horizon-bucket/robot_lab2/datasets/all_data/robotwin{1.0,2.0/...}` |
| RoboTwin 仓库本地副本 | `/home/users/kun01.wu-labs/git_repo/robotwin/`（rsync 自 xuewu.lin 的 bucket 目录，**不改原件**） |

落点规则遵循 [`../robodojo_pipeline/00_storage_layout.md`](../robodojo_pipeline/00_storage_layout.md)：
代码留 `/home`、会变的训练状态放 JFS、定版产物放 bucket。

**注意**：评测结果归档里**不含 episode 视频**。原始 800 段 mp4 只在 AIDI 侧
`output/<task>/demo_clean/episode*.mp4`，归档有留存期，到期即不可再取。
归档时的取舍见该目录下的 `README.md`。

---

## 复现（如果要重跑）

```bash
cd /home/users/kun01.wu-labs/git_repo/robo_orchard_lab
source robo_orchard_lab_env.sh          # 必须，否则缓存和 TMPDIR 回落到 98% 满的 /

# 只改 job_name 和 --model_config 两处即可换 checkpoint
RoboOrchardJob-AIDISubmit submit_from_config \
    --config projects/holobrain_internal/common/aidi_submit_config/submit_cfg_robotwin_eval_kun_mydocker.json
```

镜像 `docker.hobot.cc/imagesys/kun01.wu/holobrain-eval:ubuntu22.04-gcc11.4-py3.10-cuda12.8-torch280-robotwin-20260724-v3`
已验证可用（含 curobo）。一次约 3 小时 / 8 卡。

**不要走本机路线** —— `envs/robot/robot.py:15` 硬依赖 curobo，本机装不上，详见 [03_eval.md](03_eval.md) §1。
