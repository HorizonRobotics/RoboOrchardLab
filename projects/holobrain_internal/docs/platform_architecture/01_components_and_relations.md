# 01 — 组件与相互关系

本篇是名词表 + 拓扑图。搞清楚每个组件**是什么、在哪里、我在它上面能做什么、它和别人怎么衔接**，后面 02-05 的具体流程才有落点。

---

## 1. Bucket（DMP / horizon-bucket）

**是什么**：Horizon 内部的对象存储系统（DMP = Data Management Platform），通过 fuse 挂载暴露成 POSIX 目录 `/horizon-bucket/<bucket_name>/…`。

**在哪里可见**：
- dev machine 上：已挂载，`ls /horizon-bucket/` 就能看到 `robot_lab/ robot_lab2/ ...`
- AIDI 集群 pod 里：**必须在 `submit_cfg.json` 的 `input_bucket` / `output_bucket` 字段声明**，才会被挂载

**bucket 的读写规则**（血泪教训）：
| 路径 | 谁可写 |
|---|---|
| `/horizon-bucket/<bucket>/` 根 | **不可写**，`mkdir` 会 Permission denied |
| `/horizon-bucket/<bucket>/users/` | **不可写** |
| `/horizon-bucket/<bucket>/users/<你的 bucket 用户名>/` | **只有你自己可写** |
| 别人的 `users/<others>/` | 通常只读，别写 |
| `/horizon-bucket/<bucket>/datasets/` | 团队共享数据，通常只读，别乱改 |

**注意用户名映射**（我踩过）：
- Linux 用户名：`kun01.wu-labs`
- Bucket 用户名：`kun01.wu`（去掉 `-labs`）
- SSO 用户名 / AIDI user_name：`kun01.wu-labs`

所以我的写目录是 `/horizon-bucket/robot_lab/users/kun01.wu/` —— 参见 memory `[[kun-wu-bucket-workspace]]`。

**常用 bucket**（HoloBrain 项目相关）：
- `robot_lab` — 用户目录 + 部分数据集
- `robot_lab2` — 大型数据集（RoboDojo LMDB / all_data）

**能干什么**：
- 存 checkpoint（训练产物，deploy package）
- 存数据集（团队共享 / 自己临时）
- 存 eval 结果（`aidi_output/<job>/eval_result/`）
- **跨 dev ↔ pod 共享文件的唯一稳定通道**（dev 和 pod 都 fuse 挂载同一个 bucket）

---

## 2. DevMachine（开发机）

**是什么**：你个人的 Linux 机器（本例 `kun01.wu-labs`），跑 shell / conda / docker / IDE，是所有工作的起点。

**关键目录**：
| 路径 | 用途 |
|---|---|
| `~/git_repo/robo_orchard_lab` | 本仓库（HoloBrain 训练代码） |
| `~/git_repo/RoboDojo/` | RoboDojo 仿真评测代码（外部仓库的本地副本） |
| `~/git_repo/robo_orchard_core/` | RoboOrchard 核心库源代码 |
| `~/miniconda3/envs/holobrain_internal` | 提交 job 用的 conda env（含 `RoboOrchardJob-AIDISubmit`、`aidictl`、`aidisdk`） |
| `~/miniconda3/envs/holobrain` | 本地 policy 推理 env（评测 policy server 端；仅在容器内评测时才关键） |
| `~/miniconda3/envs/RoboDojo` | 本地 sim env（仅在容器内评测时才关键） |
| `~/.aidisdk/config.yaml` | AIDI 平台的鉴权 token（`aidictl` 和 REST API 都读它） |
| `~/.docker/config.json` | Docker Registry 登录凭证 |
| `/horizon-bucket/...` | fuse 挂载的所有 bucket |

**能干什么**：
- **提交 AIDI job**：`RoboOrchardJob-AIDISubmit submit_from_config --config <json>`
- **查 job 状态 / 拉 log**：`aidictl job status <id>` / REST API
- **准备 deploy package**：把训练完的 accelerate state 转成 deploy 格式后 `cp` 进 bucket
- **本地 docker 迭代**：`docker pull` / `docker run` / `docker commit` / `docker push`
- **本地 smoke test**：小数据 / 小 batch 跑一遍，验证代码
- **不能**跑 8 卡训练（dev 机通常只有 0-1 张卡）—— 训练必须上集群

---

## 3. AIDI 集群

**是什么**：Horizon 内部的 K8s 任务调度平台，你把 job 描述 POST 给 `/computing-apiserver/job/create`，它给你分 pod、拉镜像、mount bucket、跑你的 shell。

**关键概念**：

| 概念 | 说明 |
|---|---|
| Queue | 集群队列，决定机型/地域/项目权限；常用 `project-5090-robot-lab-bcloud-bj`（RTX 5090） |
| Project | 计费单位，本项目用 `horizon-labs` |
| Job | 一次提交 = 一个 job，得到 `bcloud-bj-zone1-<uuid12>` 形式的 ID |
| Pod / Worker | 一个 job 可以要多个 pod（`num_workers`），每 pod 挂 N 张 GPU（`gpu_per_worker`，≤8） |
| Rank | 分布式训练时 pod 0 是 rank 0，`get_rank.py` 用来分配 |
| wall_time | job 最长运行时间，**单位是分钟**（源码 typo 写成 "minitus"），到点 SIGTERM |
| Phase | job 状态：Queuing / Running / Succeeded / Failed / Cancelled |
| priority | 1-5，5 是最高。`aidictl job urgent` 是把优先级抬到 5 |

**pod 里的固定路径**：

| 路径 | 用途 |
|---|---|
| `$WORKING_PATH` | 解压后的 `workspace_folder` 位置（你 `to_upload` 的所有东西都在这里） |
| `/job_data` | 官方给你写「训练产物」的目录，job 结束会归档到 `output/` |
| `/job_tboard` | TensorBoard 日志目录，会归档到 `tboardlog/` |
| `/running_package/code_package/` | 早期版本的 `$WORKING_PATH` 别名，见 `robodojo_pipeline/03_eval.md` |
| `/horizon-bucket/<bucket>/` | 你在 `input_bucket`/`output_bucket` 里声明的 bucket 都会 fuse 挂载 |
| `/opt/miniconda3/` | 镜像里预装的 miniconda |

**能干什么**：
- 跑训练（用你镜像里的 conda env + accelerate）
- 跑评测（IsaacSim / RoboTwin / LIBERO 等 headless）
- 拉数据处理 / labeling job

**不能干什么**：
- 交互式 shell（pod 是一次性的）
- 长时间的服务（wall_time 硬限制）
- 直接访问外网（要走 hobot mirror；HuggingFace 也不通，靠 `HF_HUB_OFFLINE=1`）

**平台入口**：
- Web: (内部 URL, `aidictl job logs url` 出来的地址)
- CLI: `aidictl` (skill: [[aidi-ctl]])
- SDK: `aidisdk` (skill: [[aidi-cloud-submit]] §2.4)
- REST: `http://computing.aidi.hobot.cc/infra/api/v1alpha/computing-apiserver/` （无缓存，最新状态用这个）

---

## 4. Docker Registry & Image

**是什么**：Harbor（`docker.hobot.cc`，也写 `hub.hobot.cc`），Horizon 内部的私有 registry。集群 pod 只能拉这里的镜像。

**镜像命名规范**：

| Namespace | 用途 |
|---|---|
| `docker.hobot.cc/imagesys/robotlab-mani:...` | 公共基础镜像（cu128 + torch 2.8 + NCCL 等） |
| `docker.hobot.cc/imagesys/kun01.wu/*` | 你个人的镜像（`imagesys/<你的 SSO 名>/`） |
| `docker.hobot.cc/imagesys/<team>/*` | 团队镜像 |

**能干什么**：
- 从基础镜像起 dev container：`docker run -it ... robotlab-mani:...`
- 在 container 里装依赖：`pip install / apt install`
- `docker commit + docker tag + docker push` → 迭代出 `imagesys/<你>/<image>:vN` → 集群 `docker_image` 字段填这个

**HoloBrain 常用镜像**：
- 训练镜像：`docker.hobot.cc/imagesys/robotlab-mani:ubuntu2204-gcc11.4-cu128-nccl2277-torch280-erdma-trasnformers5102`（含 accelerate + transformers 5.10.2；tag 末尾的 "trasnformers" 是拼写错误但已固化）
- 评测镜像：`docker.hobot.cc/imagesys/kun01.wu/robodojo-holobrain:ubuntu22.04-gcc11.4-cu128-torch280-holobrain-20260727-v6`（含 IsaacSim + `holobrain` + `RoboDojo` 两个 conda env）

**关键约定**：
- **代码不烧进镜像**。代码通过 `to_upload` 每次 submit 时 rsync 上去，方便迭代。
- 依赖（conda env / pip 包 / 系统 lib）**才**进镜像。
- 镜像 tag 从不覆盖（`v6` 就是 `v6`），需要升级就 `v7`、`v8`。见 [[internal-docker]] skill。

---

## 5. HoloBrain（模型 / policy）

**是什么**：本项目训练的 **VLA (Vision-Language-Action) policy 模型**，backbone 是 Qwen2.5-VL-3B-Instruct，加了 diffusion action head，输出 joint / EE 空间的 action chunk。

**代码位置**：
- 训练入口：`projects/holobrain_internal/common/train.py`
- 配置：`projects/holobrain_internal/common/configs/config_holobrain_common.py`
- 模型：`robo_orchard_lab/models/holobrain/*`（`HoloBrain_Qwen2_5_VL`, `HoloBrainProcessor`, `HoloBrainActionLoss`, `HoloBrainInferencePipeline`）
- 部署适配（RoboDojo 侧）：`RoboDojo/XPolicyLab/policy/HoloBrain/{deploy.yml, model.py, deploy.py, setup_eval_policy_server.sh}`

**产物两种形态**（见 memory [[holobrain-checkpoint-layouts]]）：

| 形态 | 目录内容 | 用途 |
|---|---|---|
| **Accelerate state** | `checkpoint_N/{model.safetensors, optimizer.bin, scheduler.bin, random_states_0.pkl, ...}` | 只能用来 resume 训练 |
| **Deploy package** | `checkpoint_N/{model.safetensors, model.config.json, <sim>_processor.json, <sim>_inference.config.json, urdf/, ckpt→...}` | 部署 / eval 用（`--model_config` 直接吃） |

从前者转后者需要**手工 export**（见 `../09_export_and_eval.md`）——训练不会自动生成 deploy package。

**HoloBrain 的评测**：
- 不在 robo_orchard_lab 里评（本仓库只训练）
- 评测靠外部 sim 项目（RoboDojo / RoboTwin / LIBERO / RoboCasa / Isaac / Behavior1K），走「policy server + env client」的架构，见 [04_dual_env_client_server.md](04_dual_env_client_server.md)

---

## 6. RoboOrchard 全家桶

三个独立 Python 包，同一 org，功能分层：

| 包 | 用途 | 你会 import 的 | 装在 |
|---|---|---|---|
| `robo_orchard_core` | 基础库：数据结构、config 框架、日志 | `robo_orchard_core.config.*` | dev + 训练镜像 + 评测镜像 |
| `robo_orchard_lab` | 训练库：dataset、model、pipeline、hooks | `robo_orchard_lab.models.holobrain.*` / `.pipeline.*` / `.dataset.*` | dev + 训练镜像 + 评测镜像（policy 端 needs） |
| `robo_orchard_jobs` | Job 提交库：AIDI submit + jinja 模板 | 只在 dev 用；给 `RoboOrchardJob-AIDISubmit` 提供 | dev 提交 env (`holobrain_internal`) |

三者都在 `~/git_repo/` 或 conda env 里。**训练/评测镜像不需要 `robo_orchard_jobs`**（那是提交侧的东西）。

**关键 CLI**：`RoboOrchardJob-AIDISubmit`（来自 `robo_orchard_jobs.job_submit.aidi.submit_cli`）
- `submit_from_config --config X.json` — 主入口，dev 上敲
- `aidisdk_job_submit --job_config_path Y.yaml --queue_name Z --job_type T` — 内部子命令，由 `submit_from_config` 自动调，一般不手敲

---

## 7. RoboDojo（评测 benchmark）

**是什么**：外部（合作方 xiaomi robotics 主导）的仿真 benchmark，54 个双臂 tabletop 操作 task，基于 IsaacSim 4.5 + IsaacLab。

**代码位置**：
- 官方源：`/horizon-bucket/robot_lab/users/xuewu.lin/gitlab/robotwin/...`（**别改**！）
- 我的本地副本：`~/git_repo/RoboDojo/`（memory [[robotwin-repo-local-copy]] 强调所有 patch/staging 都在这里做）

**评测所需的两个 conda env（在集群镜像里）**：

| Env | Python | 关键依赖 | 谁用 |
|---|---|---|---|
| `holobrain` | 3.11 | torch 2.8/cu128, transformers 5.10.2, robo_orchard_core + lab | policy server (`XPolicyLab/policy/HoloBrain/setup_eval_policy_server.sh`) |
| `RoboDojo` | 3.10 | IsaacLab, IsaacSim 4.5, sapien 3.0.3, mplib 0.2.1, numpy<2.0 | env client (`src/eval_client/main.py`) |

两个 env 通过 **WebSocket** 通信（`XPolicyLab/client_server/ws/`），因为 IsaacSim 4.5 硬绑 py3.10，与新版 transformers 不兼容。详见 [04_dual_env_client_server.md](04_dual_env_client_server.md)。

---

## 8. 相关但独立的组件

| 组件 | 作用 | 与 HoloBrain 的关系 |
|---|---|---|
| `robotwin/` | RoboTwin benchmark（另一个 sim，非 RoboDojo） | 评测 HoloBrain 的另一条 pipeline；`submit_cfg_robotwin_eval*.json` |
| LIBERO / RoboCasa / GenieSim3 / Behavior-1K / IsaacLab | 更多评测 benchmark | 各自 `submit_cfg_<sim>_eval.json` |
| `RoboOrchardJob-AIDISubmit` vs `submit/submit.py` | 两条独立的提交路径 | 本项目用前者；`road-model` 系用后者，别混（见 [[aidi-cloud-submit]] 顶部表格） |
| `aidictl` | AIDI 平台官方 CLI | 查 job / 拉 log / 管队列，dev 侧工具 |
| `aidi-inf-cli` | 老版 AIDI CLI，`use_aidisdk: false` 时用 | 新流程都用 aidisdk，不管这个 |

---

## 9. 组件间数据流总结表

| 流向 | 走什么 |
|---|---|
| Dev → 集群 pod（代码） | `to_upload` → rsync → workspace_folder → 加密 tar → OSS → pod 拉解 |
| Dev → 集群 pod（数据） | bucket 挂载（`input_bucket`），pod 里 `ln -s /horizon-bucket/... assets` |
| Dev → 集群 pod（镜像） | Harbor pull |
| 集群 pod → bucket（训练产物） | pod 里代码 `save_checkpoint()` 直接写到 fuse 挂载的 bucket 路径；或写 `/job_data` 后 AIDI 回收到 `output/` |
| 集群 pod → 用户看 log | `aidictl job logs tail/cat/download` 或 web UI |
| Bucket → dev（拉 checkpoint） | `cp` / `rsync` from `/horizon-bucket/...`（fuse 挂载） |
| Bucket → 集群 pod（数据 / ckpt） | 同上，直接读 `/horizon-bucket/...` |
| Dev → Docker Registry（镜像） | `docker push` |
| Docker Registry → 集群 pod | AIDI 自动拉（根据 `docker_image` 字段） |

**重点**：pod 和 dev 通过 bucket **异步、间接**通信；从不直连（除非通过 log CLI）。

---

## 10. 提交路径全景（补 README 的图）

```
                                                submit_cfg_*.json  ← 你写这一份
                                                        │
                                                        ▼
   ┌────────────  DEV MACHINE  ──────────────────────────────────────┐
   │                                                                  │
   │  RoboOrchardJob-AIDISubmit submit_from_config --config X.json    │
   │        │                                                          │
   │        ▼                                                          │
   │  robo_orchard_jobs.job_submit.aidi.job_config                     │
   │        .JobSubmitParamForAIDI._command_impl():                    │
   │     1. prepare_workspace()   ── clear_workspace? rsync to_upload  │
   │     2. 拷 3 个 dist util (get_rank.py, ssh_launcher.py, url2IP.py)│
   │     3. jinja render run.sh + run_local.sh into workspace_folder/  │
   │     4. 写 job_config.yaml                                         │
   │     5. exec `RoboOrchardJob-AIDISubmit aidisdk_job_submit ...`    │
   │        （aidisdk 内部：tar → OSS → POST job/create）              │
   │                                                                   │
   │  副产品：./aidi_job_submit.json    ← 快照，可复现提交              │
   │           workspace_folder/       ← 打包源                        │
   └───────────────────┬───────────────────────────────────────────────┘
                       │
                       ▼
   ┌────────────  AIDI  ─────────────────────────────────────────────┐
   │  scheduler 分 pod, mount bucket, docker pull, 解密 tar          │
   │       →  pod:$WORKING_PATH/{run.sh, run_local.sh, <你的代码>}   │
   │       →  bash run.sh   （多 pod: get_rank + ssh_launcher）      │
   │            └─ bash run_local.sh                                 │
   │                 └─ <你 cmd 里的每一行> + <你的 python launcher>  │
   └─────────────────────────────────────────────────────────────────┘
```

下一篇 [02_end_to_end_workflow.md](02_end_to_end_workflow.md) 把这个图展开成时间线。
