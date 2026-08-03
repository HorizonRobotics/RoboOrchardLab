# HoloBrain × RoboDojo Pipeline — 完整技术教程

本文档目录记录 **HoloBrain 模型在 RoboDojo benchmark 上的完整代码通路**：从 AIDI 训练任务提交，到集群侧 accelerate 训练，到 checkpoint export 成部署包，再到集群侧 Isaac Sim 评测。所有内容都精确到文件路径、函数名、关键行号、输入输出的 shape 与 dtype。

**语境**：本教程是 kun01.wu 在 2026-07-27 到 07-28 期间实操跑通 v6 docker image + 20k/100k 训练 + seed0 sanity/full eval 的实录整理，可作为「重跑 / debug / 改动 embodiment / 换 benchmark」的手册。

> **⚠️ 2026-08-03 合并 `feature/sem_internal` 之后，有两处变更会影响本文的准确性**：
> ① 仓库默认配置已从 v9 切到 v10（VLM 换成 Qwen3.5-2B、`patch_size` 28→32），本文所有
> `config_holobrain_common.py:<行号>` 引用都已漂移，对照表见
> [`../04_config_system.md`](../04_config_system.md) 顶部；
> ② robodojo 的 processor 导出名改成了 `robodojo_arx_x5a_processor`，见
> [02_deploy_package.md](02_deploy_package.md) 顶部。
> **本文记录的仍是 v9 那次的实况，没有改写。**

> **想直接看结果** → [07_results.md](07_results.md)；**想知道当前状态与已知坑** → [STATUS.md](STATUS.md)。
>
> **评测编排层换过一次**：现在用 in-repo `common/robodojo_eval.py`（同事 xuewu.lin 的实现），
> 不再用外部 `~/git_repo/RoboDojo/` 那套。[03_eval.md](03_eval.md) 的 §1–§2 是旧流程的历史
> 记录，§3–§8（policy server / env client / wire 格式 / episode loop / result schema）
> 两套流程共用，仍然准确。

---

## 架构总览

```
┌──────────── 训练 (2 pod × 8 × RTX 5090, 8988~) ────────────┐
│                                                              │
│  submit_cfg_robodojo_train_100k.json                        │
│      │  RoboOrchardJob-AIDISubmit submit_from_config         │
│      ▼                                                       │
│  cluster pod:                                                │
│    accelerate launch projects/holobrain_internal/common/     │
│                       train.py                               │
│                       --config config_holobrain_common.py:v9│
│                       --max_step 100000 --save_step_freq 5000│
│                                                              │
│  → /job_data/checkpoints/checkpoint_{0..19}/                 │
│     (每 5k step 一份 accelerate state, total_limit=3)        │
│                                                              │
└──────────────────────────────────────────────────────────────┘
                    ▼
┌──────────── Export deploy package (手工) ──────────────────┐
│                                                              │
│  cp model.safetensors    → bucket/checkpoint_20000/          │
│  cp model.config.json    → bucket/checkpoint_20000/          │
│  cp robodojo_processor.json                                  │
│  cp robodojo_inference.config.json                           │
│  ln -sfn xuewu.lin/ckpt → bucket/checkpoint_20000/ckpt       │
│  cp urdf/robotwin2_dual_arm_arx_x5a.urdf → checkpoint_20000/urdf/ │
│                                                              │
└──────────────────────────────────────────────────────────────┘
                    ▼
┌──────────── 评测 (1 pod × 8 GPU, wall_time=48h) ────────────┐
│                                                              │
│  submit_cfg_holobrain_robodojo_seed0.json                   │
│      │  RoboOrchardJob-AIDISubmit submit_from_config         │
│      ▼                                                       │
│  cluster pod:                                                │
│    bash scripts/robodojo.sh benchmark \                      │
│         --policy-dir XPolicyLab/policy/HoloBrain \           │
│         --ckpt checkpoint_20000 \                            │
│         --env-cfg arx_x5_holobrain --eval-num 25             │
│    → scripts/internal/smoke_all_tasks.sh                     │
│      → for task in 54 tasks:                                 │
│          bash run_policy_eval.sh                             │
│          ├─ setup_eval_policy_server.sh  (holobrain env, GPU 0) │
│          │   └─ setup_policy_server.py                       │
│          │      └─ HoloBrain Model + WsPolicyServer          │
│          └─ setup_eval_env_client.sh    (RoboDojo env, GPU 1)│
│              └─ src/eval_client/main.py                      │
│                 └─ EvalEnv.run_eval (25 layouts × up to 1050 sim step) │
│                                                              │
│  → bucket/robodojo-holobrain-seed0/                          │
│      benchmark.log                                           │
│      smoke_results/&lt;run_id&gt;.{json,md} + logs/&lt;task&gt;.log       │
│      eval_result/RoboDojo/&lt;task&gt;/HoloBrain/.../{_result.json, episode_*.mp4} │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 文档结构

| 文件 | 内容 |
|---|---|
| [00_storage_layout.md](00_storage_layout.md) | **落点约定（权威）**：代码/训练状态/定版产物分别放哪、什么是自动的什么要手动归档、命名规则、两个已知坑 |
| [01_training.md](01_training.md) | 训练侧：AIDI submit_cfg 字段、`train.py` main、Config 加载 (`config_holobrain_common.py` v9)、Dataset (`RoboDojoLmdbDataset` + transforms)、Model forward (`HoloBrain_Qwen2_5_VL`)、Loss (`HoloBrainActionLoss`)、Checkpoint save (`SaveCheckpoint` hook) |
| [02_deploy_package.md](02_deploy_package.md) | 从 accelerate state 组装 deploy package：`model.safetensors` / `model.config.json` / `<sim>_processor.json` / `<sim>_inference.config.json` / `urdf/` 的来源与用途 |
| [03_eval.md](03_eval.md) | **（§1–§2 编排层已弃用，见该文首部说明；§3–§8 仍有效）** 评测侧：`smoke_all_tasks.sh` → `run_policy_eval.sh` → server (`setup_policy_server.py` + `HoloBrain.model.Model`) + client (`src/eval_client/main.py` + `EvalEnv`)、WebSocket 通信 obs/action dict shape、Isaac Sim episode loop、success/fail 判定、`_result.json` 写入 |
| [04_commands_cheatsheet.md](04_commands_cheatsheet.md) | 所有可复用的命令：提交 job、查询状态、拉日志、抓 checkpoint、解析 result.json、看视频。含 `RoboOrchardJob-AIDISubmit` / `aidictl` / `job/get` REST API 的模板 |
| [07_results.md](07_results.md) | **最终结果**：20k vs 100k 的 5 维度与逐任务 SR/score、Generalization 标准-vs-随机拆分、哪些差异可解读、数据完整性与结构性上限。配套 JSON 在 `results/{20k,100k}/` |
| [STATUS.md](STATUS.md) | 当前状态、已定决策、11 条已知坑、后续方向 |
| [05_troubleshooting.md](05_troubleshooting.md) | 已知坑：v6 image 60+ dep 修复记录、IsaacLab pin sed patch、rsync `-aL` 的 symlink dangling 问题、numpy < 2.0 硬 pin 冲突、submit_cfg 常见错误清单、3-strike fatigue 心法 |

---

## 关键数字速查

| Quantity | Value | Source |
|---|---|---|
| Dataset | RoboDojo LMDB, embodiment `arx_x5a`, 3 cams | `config_robodojo_dataset.py:77`; `dataset_specs_robodojo.py:55-65` |
| Global batch | 16 × 2 pods × 8 GPUs = **256 samples/step** | `submit_cfg_robodojo_train_100k.json` cmd L8-9 |
| Speed (5090) | ~275 samples/sec, ~0.93 s/step, ~26h for 100k step | 实测 `bcloud-bj-zone1-6c6f0a3cbcb9` |
| Prediction horizon | `pred_steps=64`, `chunk_size=4` → 16 action chunks | `config_holobrain_common.py:21-22` |
| Diffusion | DDPM 1000-step train / DPMSolver++ 10-step inference | `config_holobrain_common.py:380-393` |
| VLM | Qwen2.5-VL-3B-Instruct, bf16, LM trimmed to first 4 layers | `config_holobrain_common.py:86-88` |
| Loss keys | `loss_angle, loss_xyz, loss_rot, loss_angle_fk, loss_xyz_fk, loss_rot_fk` | `models/holobrain/loss.py:135-141` |
| Ckpt dir pattern | `/job_data/checkpoints/checkpoint_N/` (total_limit=3) | `pipeline/hooks/checkpoint.py:78-219` |
| Eval tasks | 54 (RoboDojo, `smoke_all_tasks.sh` + `task_inventory.py`) | 实测 seed0 job 主 log |
| Eval / task | ~30-160 min at eval_num=25, wall_time=48h → 完成 ~26-30 tasks | 实测 xiaomi + kun 的两次 seed0 |
| Success predicate | `reward > 1-1e-3` else fail at `step_lim`（arrange_largest_number: 1050） | `env/reward_manager/reward_manager.py:527`; `task/RoboDojo/tasks/*.py` |

---

## 常用命令三件套

```bash
# 1) 提交训练
cd /home/users/kun01.wu-labs/git_repo/robo_orchard_lab
RoboOrchardJob-AIDISubmit submit_from_config \
    --config projects/holobrain_internal/common/aidi_submit_config/submit_cfg_robodojo_train_100k.json

# 2) 查询 job 状态（不用 aidictl list 因为有 15min 缓存）
python3 <<'PY'
import requests
token = open("/home/users/kun01.wu-labs/.aidisdk/config.yaml").read().split("token:")[1].split("\n")[0].strip()
r = requests.get("http://computing.aidi.hobot.cc/infra/api/v1alpha/computing-apiserver/job/get",
                 headers={"Authorization": token},
                 params={"job_id": "bcloud-bj-zone1-6c6f0a3cbcb9"})
print(r.json()["data"]["job_status"]["phase"])
PY

# 3) 拉最新 loss / task 进度
aidictl job logs tail bcloud-bj-zone1-6c6f0a3cbcb9 log/bcloud-bj-zone1-6c6f0a3cbcb9-task-1-main.log \
    | grep "GlobalStep" | tail -5
```

更多在 [04_commands_cheatsheet.md](04_commands_cheatsheet.md)。

---

## 依赖关系与前置知识

1. **repo 组织**：`robo_orchard_lab` 是主训练库（本仓库根）；`RoboDojo` 是外部仿真库（`~/git_repo/RoboDojo/`，包含 IsaacLab + 54 task 定义 + XPolicyLab client-server 框架）；两者在评测时通过 `to_upload` 把 `robo_orchard_lab/robo_orchard_lab/` **真实拷贝**（不能是 symlink）打进 `/running_package/code_package/`。
2. **两个 conda env**：
   - `holobrain` — policy server 端，装了 torch 2.8/cu128 + transformers 5.10.2 + robo_orchard_core + pytorch3d CPU-only + pytorch_kinematics
   - `RoboDojo` — env client 端，装了 IsaacLab + IsaacSim 4.5 + sapien 3.0.3 + mplib 0.2.1
   - 两 env 通过 WebSocket 通信 (`XPolicyLab/client_server/ws/`)，避免 Isaac Sim 4.5 强绑 py3.10 与 transformers 新版冲突。
3. **Bucket 布局**（见 memory [[kun-wu-bucket-workspace]]）：
   - `/horizon-bucket/robot_lab/users/kun01.wu/datasets/RoboDojo/Assets/` — RoboDojo Assets（含 `Eval_Layout/RoboDojo/arx_x5_holobrain -> ../arx_x5` symlink）
   - `/horizon-bucket/robot_lab/users/kun01.wu/robo_orchard_lab/` — **本项目产物根**
     （2026-07-30 归集，与 `univtac/` `xiaomi_robodojo/` 同级）：
     `ckpts/holobrain_robodojo_posttrain_v9{,_100k}/` 训练产出的 deploy package、
     `ckpts/checkpoint_11_eval/` robotwin 期组装目录、
     `eval_results/robodojo-holobrain-{eval-final,sanity,seed0}/` 评测产物
   - `/horizon-bucket/robot_lab/users/kun01.wu/aidi_output/` — AIDI 按 `job_name` 自动同步
     `/job_data` 的落点，**仍是新训练 job 的第一落点**，跑完需自行归集到上面的项目根
   - `/horizon-bucket/robot_lab/users/xuewu.lin/ckpt` — Qwen VLM base weight 的父目录（HoloBrainProcessor.load 相对路径解析用）
   - `/horizon-bucket/robot_lab2/datasets/all_data/robodojo/lmdb/*` — RoboDojo LMDB 训练数据

---

## 记录的实测参数

| 场景 | Job ID | 结果 |
|---|---|---|
| 5k sanity 训练 | `bcloud-bj-zone1-4fb0ee2ff3d4` | Succeeded, 验证 pipeline |
| 20k 训练 (baseline) | `bcloud-bj-zone1-1f00b8e23ac8` | Succeeded, `total_loss=0.098` @ step 19999, 20k eval SR=0 |
| 100k 训练 | `bcloud-bj-zone1-6c6f0a3cbcb9` | Running, `total_loss=0.07` @ step 58k |
| sanity smoke eval (2 task) | `bcloud-bj-zone1-805a64eaab5f` | PASS, 15 min/task |
| full seed0 eval (54 task × 25 ep) | `bcloud-bj-zone1-7895445e92bc` | Running, 6/54 task PASS (SR=0.0 for 20k ckpt) |

