# Platform Architecture — Bucket / DevMachine / AIDI / Docker / HoloBrain 整体图景

> 面向新人 / 短期回坑者的**一体化平台知识手册**。回答三个层次的问题：
>
> 1. 这些概念是什么，它们之间的关系？
> 2. 一次训练/评测在这些组件之间是怎么流动的？
> 3. 遇到 `submit_cfg*.json` 里的字段、双 conda env、集群 pod 里跑什么 —— 具体怎么对上号？
>
> 已有的 `robodojo_pipeline/` 是「某一次任务的深挖实录」，本目录是「跨任务的平台性总览」，两者互补。RoboDojo 的具体细节仍以 `robodojo_pipeline/` 为准。

---

## 文档目录

| 文件 | 内容 |
|---|---|
| [01_components_and_relations.md](01_components_and_relations.md) | Bucket / DevMachine / AIDI 集群 / Docker Registry / HoloBrain / RoboOrchard 全家桶 / RoboDojo 等组件的**定义、职责、边界、相互关系**。这是名词表 + 拓扑图。 |
| [02_end_to_end_workflow.md](02_end_to_end_workflow.md) | **端到端工作流**：从开发机敲下 `submit_from_config`，到集群 pod 拉镜像、rsync 代码、执行 `run.sh`、accelerate 分布式、checkpoint 落 bucket、评测 rsync 结果回来的完整时间线。 |
| [03_submit_cfg_lifecycle.md](03_submit_cfg_lifecycle.md) | **具体问题 (1)**：以 `submit_cfg.json` 为例，任务进哪个目录、什么时候激活 conda、跑什么程序、每个字段映射到集群侧什么行为、**默认值来自哪里怎么看**。 |
| [04_dual_env_client_server.md](04_dual_env_client_server.md) | **具体问题 (2)**：为什么评测要两个 conda env（`holobrain` + `RoboDojo`），怎么在一个 pod 里既跑 policy server 又跑 env client，用什么协议、什么端口、怎么互斥 GPU、怎么优雅收尾。 |
| [05_faq_and_hidden_gotchas.md](05_faq_and_hidden_gotchas.md) | **具体问题 (3) + 你没想到的**：`aidictl list` 缓存、rsync `-aL` symlink 坑、workspace 撑爆、SR=0 但 loss 好看、僵尸 job 消耗集群等，附诊断命令。 |

---

## 一分钟拓扑图

```
                                                             ┌───────────────────────┐
                                                             │  Docker Registry      │
                                                             │  docker.hobot.cc      │
                                                             │  (Harbor)             │
                                                             │  imagesys/robotlab*   │
                                                             │  imagesys/kun01.wu/*  │
                                                             └────────▲──────────────┘
                                                                      │ (pod 拉镜像)
              ┌──────────────────────────────┐                        │
              │ DevMachine (kun01.wu-labs)   │                        │
              │   ~/git_repo/robo_orchard_lab│                        │
              │   ~/git_repo/RoboDojo/       │                        │
              │   ~/miniconda3/envs/         │                        │
              │     holobrain_internal ⬅ 提交用│                       │
              │   ~/.aidisdk/config.yaml     │                        │
              │                              │                        │
              │  submit_from_config          │                        │
              │  ──────────►                 │                        │
              │  1) rsync -aL to_upload → workspace_folder/           │
              │  2) 生成 run.sh / run_local.sh / job_config.yaml      │
              │  3) 加密 tar，上传 OSS                                 │
              │  4) POST /computing-apiserver/job/create              │
              └──────────────┬───────────────┘                        │
                             │                                        │
                             ▼                                        │
              ┌──────────────────────────────────────────────┐        │
              │           AIDI 集群 (bcloud-bj)              │        │
              │  Queue: project-5090-robot-lab-bcloud-bj     │        │
              │  Project: horizon-labs                       │        │
              │                                              │        │
              │  ┌──────── Pod 0 (rank 0) ─────────┐         │        │
              │  │ /running_package/code_package/   │◄────────────────┘
              │  │   （解密 tar，即 workspace_folder ）│
              │  │ /job_data      → checkpoints/    │
              │  │ /job_tboard    → tensorboard     │
              │  │ /horizon-bucket/... (fuse mount)│──────┐
              │  │                                  │      │
              │  │ bash run.sh                      │      │
              │  │   → bash run_local.sh            │      │
              │  │     → user cmd (accelerate ...)  │      │
              │  └──────────────────────────────────┘      │
              │  ┌──────── Pod 1 (rank 1) ─────────┐       │
              │  │  同 Pod 0，get_rank.py 拿 rank    │      │
              │  └──────────────────────────────────┘      │
              └────────────────────────────────────────────┘
                                                            │
              ┌──────────────────────────────────────────────▼────┐
              │  Bucket (DMP / OSS, fuse 挂载到 pod 和 dev)         │
              │  /horizon-bucket/robot_lab/                        │
              │    users/kun01.wu/  (仅自己可写)                    │
              │      robo_orchard_lab/   ← 本项目产物（2026-07-30 归集）│
              │        ckpts/holobrain_robodojo_posttrain_v9/      │
              │          checkpoint_20000/ … deploy package        │
              │        eval_results/                               │
              │      datasets/  ckpts/  aidi_output/  ← 跨项目共用   │
              │  /horizon-bucket/robot_lab2/                       │
              │    datasets/all_data/  (只读大数据)                │
              └────────────────────────────────────────────────────┘

              「HoloBrain」= 训练的 policy 模型（本仓库的核心产物）
              「RoboOrchardJob-AIDISubmit」= 从 dev 上向 AIDI 投任务的 CLI（robo_orchard_jobs 包）
              「RoboDojo」= 外部的 IsaacSim benchmark，评测时用
              「aidictl / aidisdk」= AIDI 平台的官方 CLI / SDK，dev 上装了才能查 job
```

---

## 阅读顺序建议

- **完全新手**：01 → 02 → 03 → 04 → 05，按顺序读。
- **只想搞清楚 submit_cfg 里的字段**：直接跳 03，遇到不认识的组件回 01 补。
- **只想跑评测**：02 (workflow) + 04 (双 env) + `robodojo_pipeline/03_eval.md`。
- **在 debug**：先 05 (gotchas)，再回 03/04 定位具体机制。

---

## 已存在的相关文档索引

`../` 里的其他 docs：

- `01_overview.md` — HoloBrain 项目自身介绍（模型/motivation）
- `02_repo_structure.md` — 本仓库代码结构
- `03_env_and_quickstart.md` — dev 机 conda env 初始化
- `04_config_system.md` — HoloBrain 配置系统 (`config_holobrain_common.py`)
- `05_dataset_pipeline.md` — 数据集
- `06_model_architecture.md` — 模型结构
- `09_export_and_eval.md` — export deploy package + eval
- `robodojo_pipeline/*.md` — 一次跑通 HoloBrain-on-RoboDojo 的完整实录（本目录不重复，遇到时链接）
- `claude_tasks/*.md` — session handoff 备忘

---

## 术语速查

| 缩写 | 全称 / 含义 |
|---|---|
| AIDI | Horizon 内部的机器学习平台（Kubernetes-based 任务调度） |
| DMP | Data Management Platform，即 bucket 系统 |
| bcloud-bj | 北京 zone1 集群，job_id 形如 `bcloud-bj-zone1-xxx` |
| pod | AIDI 分配的 K8s pod，一个 pod ≈ 一台机器（NxGPU） |
| workspace_folder | dev 侧临时目录 + 集群 pod 里的 `/running_package/code_package/` |
| WORKING_PATH | 集群 pod 内的环境变量 = 解压后的 code_package 路径 |
| /job_data | 集群 pod 内绑定给你写「训练产物 / eval 输出」的目录（会被回收前 rsync 到 bucket） |
| /job_tboard | 集群 pod 内的 TensorBoard 目录 |
| run.sh / run_local.sh | AIDI 侧的 entrypoint，由 `robo_orchard_jobs` 用 jinja 模板生成，最终执行你 `cmd` 里的内容 |
| RoboOrchard 全家桶 | `robo_orchard_core` + `robo_orchard_lab` + `robo_orchard_jobs` 三个包 |
