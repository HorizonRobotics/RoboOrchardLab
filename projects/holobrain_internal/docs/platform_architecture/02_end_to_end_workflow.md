# 02 — 端到端工作流

一次「敲命令 → 拿到结果」的完整时间线。分**训练**和**评测**两条主线，最后合并到「产物在哪里、怎么监控」。

---

## 主线 A：训练（HoloBrain on RoboDojo LMDB）

以 `submit_cfg_robodojo_train_100k.json` 为例。

### 阶段 0 — Dev 侧准备（只做一次 / 迭代时改）

1. **确认镜像可用**
   ```bash
   docker pull docker.hobot.cc/imagesys/robotlab-mani:ubuntu2204-...-trasnformers5102
   ```
   镜像 tag 直接写进 `submit_cfg.json:docker_image`。若要加装 dep：起 dev container → `pip install` → `docker commit` → 新 tag → `docker push` → 改 submit_cfg。见 [[internal-docker]] skill。

2. **确认 bucket 数据到位**
   ```
   /horizon-bucket/robot_lab2/datasets/all_data/robodojo/lmdb/          ← 训练数据
   /horizon-bucket/robot_lab2/datasets/all_data/urdf/urdf_v20260711/    ← URDF
   /horizon-bucket/robot_lab/users/xuewu.lin/ckpt/                       ← Qwen VLM base weight
   ```
   （HoloBrain 训练需要这三个，都由 submit_cfg 的 `cmd` 段 `ln -s ... assets` 拉进 pod）

3. **进 dev 提交 env**
   ```bash
   source ~/miniconda3/etc/profile.d/conda.sh
   conda activate holobrain_internal
   which RoboOrchardJob-AIDISubmit    # 应该能找到
   ```

### 阶段 1 — Dev 侧执行提交

```bash
cd ~/git_repo/robo_orchard_lab
RoboOrchardJob-AIDISubmit submit_from_config \
    --config projects/holobrain_internal/common/aidi_submit_config/submit_cfg_robodojo_train_100k.json
```

这一条命令背后：

| 步骤 | 发生了什么 | 代码位置 |
|---|---|---|
| a | 读 JSON → pydantic model `JobSubmitParamForAIDI` | `robo_orchard_jobs/job_submit/aidi/job_config.py:165` |
| b | 打印所有字段值到 stdout；写快照到 `./aidi_job_submit.json` | `job_config.py:283-315` |
| c | 若 `clear_workspace=true` 且 `workspace_folder` 存在 → 删掉整个目录 | `job_config.py:264` |
| d | `mkdir workspace_folder/` | `job_config.py:267` |
| e | 对 `to_upload` 每一项：`rsync -aL <path> workspace_folder/`（symlink follow） | `job_config.py:278` |
| f | 把 3 个分布式工具 (`get_rank.py`, `ssh_launcher.py`, `url2IP.py`) rsync 进 workspace | `job_config.py:352-371` |
| g | jinja 渲染 `run.sh`（多 pod 分支 vs 单 pod 分支）到 `workspace_folder/run.sh` | `job_config.py:380` |
| h | jinja 渲染 `run_local.sh`（把你 `cmd` + `python_executable` 拼成完整脚本）到 `workspace_folder/run_local.sh` | `job_config.py:389` |
| i | 生成 `workspace_folder/job_config.yaml`（给 aidisdk 用的 kube-friendly YAML） | `job_config.py:373` |
| j | subprocess.check_call 调 `RoboOrchardJob-AIDISubmit aidisdk_job_submit --job_config_path Y.yaml --queue_name Z --job_type T` | `job_config.py:404` |
| k | aidisdk 内部：加密 tar workspace → 上传 OSS → POST `/computing-apiserver/job/create` → 返回 `job_id` | `aidisdk_job_submit.py` |
| l | 打印 `Command executed: RoboOrchardJob-AIDISubmit aidisdk_job_submit ...` 后 return | `job_config.py:407` |

**Dev 端到此结束**。⚠️ 打印 `Command executed:` **不代表 job 创建成功**，只代表 subprocess 返回 0。真实 job_id 只在 aidisdk 的 INFO 日志里（默认吞了）。见 [05_faq_and_hidden_gotchas.md](05_faq_and_hidden_gotchas.md) §「AIDI SDK 三大陷阱」。

### 阶段 2 — AIDI scheduler 侧

1. 收到 `POST job/create`，验签、鉴权、入库；返回 `job_id = bcloud-bj-zone1-<uuid12>`
2. 根据 `queue_name` + `priority` 入队；等 GPU 到手 → Phase: `Queuing → Running`
3. 拉 `docker_image` 到 worker node；启 pod × `num_workers`
4. 把加密 tar 解压到每个 pod 的 `$WORKING_PATH`（即 `/running_package/code_package/`）
5. Mount `input_bucket` / `output_bucket` 到 `/horizon-bucket/<bucket>/`
6. 在每 pod 里 `bash $WORKING_PATH/run.sh`

### 阶段 3 — Pod 内执行（rank 0 视角）

`run.sh` (jinja 生成)：
```bash
set -e
cd ${WORKING_PATH}
# num_workers > 1 时：
python3 url2IP.py                              # 解 hostname → IP
python3 ssh_launcher.py --monitor --nworker 2 \
    --ngpus 8 -H /job_data/mpi_hosts \
    'bash run_local.sh'                        # ← ssh 到每个 pod 跑 run_local.sh
# num_workers == 1 时直接：
# bash run_local.sh
```

`run_local.sh` (jinja 生成)：
```bash
set -e
python3 get_rank.py --launcher accelerate      # 生成 ./rank 文件
NODE_INFO=$(cat ./rank)                        # 类似 "--machine_rank 0 --main_process_ip 10.x.x.x"
echo "NODE_INFO: $NODE_INFO"
# 然后拼上你 cmd + python_executable 的所有内容：
ulimit -n 65536
ln -s /horizon-bucket/robot_lab2/datasets/all_data ${WORKING_PATH}/data
ln -s /horizon-bucket/robot_lab2/datasets/all_data/urdf/urdf_v20260711 ${WORKING_PATH}/urdf
ln -s /horizon-bucket/robot_lab/users/xuewu.lin/ckpt ${WORKING_PATH}/ckpt
export PYTHONPATH=${WORKING_PATH}/robo_orchard_lab:$PYTHONPATH
 accelerate launch  --num_machines 2 --num-processes 16  --multi-gpu --gpu-ids 0,1,2,3,4,5,6,7  $NODE_INFO --main_process_port 1227 train.py --workspace /job_data --logging_dir /job_tboard --config configs/config_holobrain_common.py --kwargs '{"dataset_specs":"configs/dataset_specs_robodojo.py",...}'
```

**关键**：
- `cwd` 是 `$WORKING_PATH`，不是 `/tmp` 也不是 pod 根
- 你 `cmd` 里出现的 `${WORKING_PATH}` 会**在 pod 里**展开（不是 dev 侧）
- `accelerate launch` 命令行是**自动拼的**（jinja + `generate_cmd_str`），你只要写 `python_executable="train.py --arg1 --arg2"`；`--num_machines / --num-processes / --gpu-ids / $NODE_INFO / --main_process_port` 全部由 `robo_orchard_jobs.job_submit.submit_config.JobSubmitParams.generate_cmd_str()` 生成（见 [03_submit_cfg_lifecycle.md](03_submit_cfg_lifecycle.md) §「python_launcher 展开表」）

### 阶段 4 — 训练脚本 `train.py`

- accelerate 拉起 `Accelerator()`，同步 rank / seed / mixed_precision
- 读 `configs/config_holobrain_common.py`（走 robo_orchard_core 的 config 系统）
- 构建 dataset / dataloader / model / optimizer / scheduler
- 挂 hooks（`StatsMonitor`、`LossMovingAverageTracker`、`SaveCheckpoint`）
- 主循环 `SimpleTrainer.fit()`，每 `save_step_freq` 步 → `SaveCheckpoint` 写 `/job_data/checkpoints/checkpoint_N/`
- 达到 `max_step` 后正常退出

**产物在 pod 内的路径**：
- `/job_data/checkpoints/checkpoint_{0..19}/` — accelerate state（每 5k step 一份，`total_limit=3` 只留最新 3 份，见 memory `[[holobrain-checkpoint-layouts]]`）
- `/job_tboard/` — TensorBoard scalars

### 阶段 5 — Job 完成 / wall_time 到期

- AIDI 把 `/job_data` 归档到 `output/`，`/job_tboard` 归档到 `tboardlog/`，`stdout+stderr` 归档到 `log/`
- 通过 `aidictl job logs {ls,tail,cat,download} <job_id> {output,log,tboardlog}` 访问

⚠️ **重要**：`output/` 是「训练完的一次性快照」，如果你想让训练产物**边跑边落到 bucket**（避免 pod OOM 时全丢），要在 `cmd` 里主动 rsync 或让 checkpoint 直接写到 `/horizon-bucket/robot_lab/users/<你>/aidi_output/...`。HoloBrain 训练目前是写 `/job_data`，靠 AIDI 归档；评测则用 bg rsync 保险丝（[03_eval.md](../robodojo_pipeline/03_eval.md) §1.2 L26-32）。

### 阶段 6 — Dev 侧监控

```bash
# ⚠️ 15min 缓存，用 REST API 更快
JOB_ID=bcloud-bj-zone1-6c6f0a3cbcb9
aidictl job status $JOB_ID                             # phase
aidictl job logs tail $JOB_ID log/$JOB_ID-task-1-main.log | grep GlobalStep | tail
aidictl job logs download $JOB_ID output/ --dest ~/tmp/
```

见 `robodojo_pipeline/04_commands_cheatsheet.md` §2-3 拿完整命令。

### 阶段 7 — Export deploy package（手工）

训练完的 `/job_data/checkpoints/checkpoint_N/` 是 accelerate state，**不能直接给评测端用**。需要：

1. 从 bucket 里拷出 `model.safetensors`
2. 用 `HoloBrainInferencePipeline.export()` 生成 `<sim>_processor.json` + `<sim>_inference.config.json` + `model.config.json`
3. `cp` / `ln -sfn` 到 bucket 的 export 目录
4. 保证 `checkpoint_N/ckpt/` symlink 指向 Qwen VLM base weight
5. 保证 `checkpoint_N/urdf/` 有 URDF

详见 `robodojo_pipeline/02_deploy_package.md`。

---

## 主线 B：评测（HoloBrain on RoboDojo Sim）

以 `~/git_repo/RoboDojo/aidi_submit/cfgs/submit_cfg_holobrain_robodojo_seed0.json` 为例。**注意提交时 cwd 是 RoboDojo 那边**，不是 robo_orchard_lab。

### 阶段 0 — Dev 侧准备

1. **两个 conda env 必须都在评测镜像里** —— 用 `docker.hobot.cc/imagesys/kun01.wu/robodojo-holobrain:...v6`（见 memory `[[robodojo-holobrain-eval-image-v6]]`）
2. **deploy package 已经在 bucket**（阶段 A.7 已完成）
3. **`~/git_repo/RoboDojo/` 下的 `robo_orchard_lab/` 必须是实拷贝**（不能是 symlink），因为 AIDI rsync `-aL` 会 dangling
4. **bucket 侧 symlink** 已建好（见 `05_troubleshooting.md` §5）

### 阶段 1-3 — 提交、调度、pod 启动

同主线 A，唯一差别：
- `python_launcher: python3`（不是 accelerate；评测不需要分布式，policy server 单卡）
- `python_executable: null` 或不设，所有命令写在 `cmd` 里（一大坨 shell）
- `cmd` 里做几件关键事：
  - `source /opt/miniconda3/etc/profile.d/conda.sh`
  - 环境变量 (`LD_LIBRARY_PATH`, `HF_HUB_OFFLINE=1`, `TMPDIR`, `DEPLOY_PROXY_HOST=127.0.0.1`)
  - 4 个 symlink (`Assets`, `urdf`, `ckpt`, `checkpoint_20000`)
  - 改 embodiment config 里的绝对路径 (`utils/update_embodiment_config_path.py`)
  - IsaacLab pin sed patch
  - `conda activate RoboDojo`（**默认激活 sim env**，policy server 在子进程里再切 `holobrain`）
  - 起后台 rsync flush 保险丝
  - **主命令**：`bash scripts/robodojo.sh benchmark ...`
  - 汇总：`python scripts/internal/summarize_result.py`

### 阶段 4 — 评测流程（pod 内）

见 [04_dual_env_client_server.md](04_dual_env_client_server.md) 的完整拆解。简化版：

```
bash scripts/robodojo.sh benchmark
  └── scripts/internal/smoke_all_tasks.sh
       └── for task in 54_runnable_tasks:
             bash scripts/robodojo.sh eval --task $task ...
              └── scripts/internal/run_policy_eval.sh
                   ├── (bg) setup_eval_policy_server.sh    ← conda activate holobrain, GPU 0
                   │        python XPolicyLab/setup_policy_server.py --config_path deploy.yml
                   │        (加载 checkpoint_20000, 起 WebSocket server on $PORT)
                   ├── wait_for_policy_server.sh $PORT 600
                   └── setup_eval_env_client.sh            ← conda activate RoboDojo (from 阶段 3 已激活), GPU 1
                            python src/eval_client/main.py --port $PORT --policy_server_url ws://localhost:$PORT
                            (启动 IsaacSim, 跑 25 episode/task)
             record_result / write_summaries
```

### 阶段 5 — 结果落 bucket

- `_result.json` 由 `EvalEnv.run_eval` 每 batch 落到 `eval_result/RoboDojo/<task>/HoloBrain/<config>/<seed>_<info>/<run_id>/_result.json`
- 视频 `episode_XXXXXXX_cam_{head,left_wrist,right_wrist}_{success,fail}.mp4` 同目录
- `smoke_results/<run_id>.json` 由 `smoke_all_tasks.sh:229` 每 task 写一次
- `_summary.md` 由 `summarize_result.py` 最后写一次
- **`eval_result/` 和 `smoke_results/` 都是 pod 里 `ln -s /horizon-bucket/... eval_result` 建的 symlink**，所以写入直落 bucket。后台 rsync 是二次保险（防 ffmpeg buffer）

### 阶段 6-7 — Dev 侧看结果

```bash
BUCKET=/horizon-bucket/robot_lab/users/kun01.wu/robo_orchard_lab/eval_results/robodojo-holobrain-seed0
find $BUCKET/eval_result -name "_result.json" | head
cat $BUCKET/eval_result/RoboDojo/_summary.md
ls $BUCKET/smoke_results/
```

命令详见 `robodojo_pipeline/04_commands_cheatsheet.md` §4。

---

## 主线 C：docker image 迭代（跨越训练/评测）

新增一个 dep（如「训练镜像里缺 pytorch_kinematics」）：

```
DEV:  1. docker pull docker.hobot.cc/imagesys/robotlab-mani:<current_tag>
      2. docker run -it --gpus all \
             -v /home/users/kun01.wu-labs/git_repo:/git_repo \
             -v /horizon-bucket:/horizon-bucket:ro \
             <image> bash
DEV(container):
      3. conda activate holobrain
      4. pip install pytorch_kinematics --index-url http://mirrors.hobot.cc/...
      5. python -c "import pytorch_kinematics"    # 验证
      6. exit
DEV:  7. docker commit <container_id> docker.hobot.cc/imagesys/kun01.wu/<image>:<current_tag>-<yy-mm-dd>-v7
      8. docker push docker.hobot.cc/imagesys/kun01.wu/<image>:...-v7
      9. 改 submit_cfg.json:docker_image → 新 tag
      10. 提交测试 job 验证
```

见 [[internal-docker]] skill + memory [[robodojo-holobrain-eval-image-v6]]（记录了从 v1 到 v6 每个 dep 的增删）。

---

## 主线 D：Dev 侧本地训练/评测（可选，跳过 AIDI）

**本地训练**：dev 机 GPU 少（0-1 张），只做 config 语法检查 + dataloader smoke。
```bash
conda activate holobrain
cd ~/git_repo/robo_orchard_lab
python projects/holobrain_internal/common/train.py \
    --workspace /tmp/holobrain_smoke \
    --logging_dir /tmp/holobrain_smoke_tb \
    --config projects/holobrain_internal/common/configs/config_holobrain_common.py \
    --kwargs '{"batch_size":1,"max_step":10,"num_workers":0}'
```

**本地评测**：起 dev container 跑 RoboDojo，见 `robodojo_pipeline/04_commands_cheatsheet.md` §11。适合 debug 单 task。

---

## 时间线速查表

| 场景 | 从敲提交到看到第一条结果 | 备注 |
|---|---|---|
| 5k step sanity 训练 | 15 min Queuing + 20 min Running = ~35 min | 见 job `4fb0ee2ff3d4` |
| 100k step 全训练 | Queuing + 26 h Running = ~30 h | ~275 samples/sec on 2×8×5090 |
| Sanity smoke eval (2 task × 1 ep) | Queuing + 30 min = ~1 h | job `805a64eaab5f` |
| Full seed0 eval (54 task × 25 ep) | Queuing + 48 h wall_time = ~50 h → 完成 ~26-30 task | 48 h 装不下 54 task，见 `05_troubleshooting.md` §9 |
| Docker image 迭代（加 1 dep 并 push） | 5-15 min（不含 push 大 layer 时间） | push v6 ~4 GB, hobot 内网 ~2 min |

---

## 生命周期看板

```
时间轴：t=0 (敲命令) ─────────────► t=job.end
├── Dev 侧 <5s: prepare_workspace + rsync + jinja + tar + upload OSS
├── Dev 侧 <30s: aidisdk POST create → 得 job_id
├── AIDI Queuing: 5min ~ 5h（看 queue free）
├── AIDI 拉镜像: 2-10 min（image 4-16 GB，node 冷缓存慢）
├── Pod boot: ~30s（mount bucket + 解 tar）
├── run.sh 启动: ~1s
├── 你 cmd 执行: 分场景（分钟到小时）
│      ├── 训练：hours
│      └── 评测：hours
├── Pod 收尾: /job_data 归档（几十秒 ~ 几分钟，看产物大小）
└── job Phase → Succeeded / Failed
```

下一篇 [03_submit_cfg_lifecycle.md](03_submit_cfg_lifecycle.md) 把 pod 里执行的每一步和 JSON 字段一一对应。
