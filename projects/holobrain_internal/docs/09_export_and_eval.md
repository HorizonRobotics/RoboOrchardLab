# 09 · 导出与评估

> **阅读前置**：[03_env_and_quickstart](./03_env_and_quickstart.md)、[06_model_architecture](./06_model_architecture.md)
>
> **本章目标**：知道每个 eval 脚本干什么、跑法长什么样、要 mock 或 shell 出去的是什么进程；能选出最合适的评估方式验证自己的 checkpoint。

---

## 9.1 `export.py`：模型打包

来源：`projects/holobrain_internal/common/export.py`。它是训练之后**把 checkpoint + processor + inference pipeline 打成自包含目录**的工具。

### 9.1.1 命令行

```bash
python3 export.py \
    --config configs/config_holobrain_common.py \
    --workspace ./exported \
    --reload_test \
    --dataset_names libero_goal,robotwin2_0_aloha_v2 \
    --kwargs '{"checkpoint": "./workspace/checkpoints/checkpoint_9/model.safetensors"}'
```

`--kwargs` 与训练时同义（JSON 字符串或文件路径），常用来**指向自己训练的 checkpoint**。

### 9.1.2 产物结构

```
exported/
├── configs/                                # 完整拷贝
├── libero_goal_processor.json              # 每个 deploy dataset 一份 processor json
├── robotwin2_0_aloha_v2_processor.json
└── model/
    ├── model.safetensors                   # ★ 权重
    ├── model.config.json                   # 用于 ModelMixin.load_model
    ├── libero_goal_inference.config.json
    ├── robotwin2_0_aloha_v2_inference.config.json
    └── urdf/                               # processor 引用到的 URDF 目录被复制进来
```

`export.py:63-115` 精确控制上面这几步：先 `build_processors(config)` 出 `{dataset_name: HoloBrainProcessor}`；`--dataset_names` 白名单过滤后逐一 `.save(workspace, "<name>_processor.json")`；再 `build_model + load_checkpoint + model.save_model(model_path)`；最后每个 processor 拼一个 `HoloBrainInferencePipeline(cfg, model).save_pipeline(model_path, inference_prefix=f"{name}_inference", save_model=False)`。

`--reload_test` 触发一次 round-trip：`HoloBrainProcessor.load(...)` + `ModelMixin.load_model(model_path, load_impl="native")` + `HoloBrainInferencePipeline.load_pipeline(...)`——验证导出物能被下游 harness 反序列化。

## 9.2 评估脚本一览

每个 eval 脚本对应一个 (sim / 真机) 环境。形状分两类：

- **subprocess 型**：shell 出到目标仿真的官方 CLI（如 `eval_policy.py`），采集 `results.json` / `log` 里的成功率。适合"仓库外仿真"。
- **进程内型**：直接 `import` 环境的 Python API 走 `HoloBrain*Policy` 完成 rollouts。适合"仓库内已装环境"。

| 脚本 | 类型 | 依赖仓库 | 主 CLI 参数 |
|------|------|----------|-------------|
| `libero_eval.py` | subprocess (per-task) | LIBERO / LIBERO-Plus | `--benchmark, --task_suite, --num_trials_per_task, --processes_per_gpu` |
| `robotwin_eval.py` | subprocess (per-task) | RoboTwin | `--task_names, --task_config, --test_num` |
| `robocasa_eval.py` | 进程内 | RoboCasa | `--tasks, --num_trials, --video_cameras` |
| `isaac_eval.py` | subprocess (per-task) | Orchard Isaac | `--task_names, --multi_task_config, --rollouts, --maximum_step` |
| `behavior1k_eval.py` | subprocess + gloo 多节点 | OmniGibson / BEHAVIOR-1K | `--instances_to_run, --instance_per_job` |
| `realworld_eval.py` | Flask WSGI 服务 | 真机客户端 | `--port, --interpolation, --clip_action_len` |
| `robochallenge_eval.py` | 长连接 online loop | RoboChallengeInference | `--mock, --user_token, --submission_id` |
| `geniesim3_inference_server.py` | asyncio WebSocket | GenieSim3 | `--host, --port, --sampling_ratio, --use_depth` |

## 9.3 `libero_eval.py`

来源：`projects/holobrain_internal/common/libero_eval.py`（511 行）。

**做的事**：
1. 通过 `libero_utils.get_benchmark_module(...).get_benchmark_dict()` 拿到全部 (suite, task_id) 组合；
2. `allocate_tasks_to_workers(...)` 按 GPU × `processes_per_gpu` 均分任务；
3. 每个 worker 是一个 `multiprocessing.Process`，把 `CUDA_VISIBLE_DEVICES=gpu_id` 后 shell 出到 `projects/holobrain_internal/libero/eval_policy.py`（policy adapter），带一堆参数（`--model_config, --model_prefix, --vlm_ckpt_dir, --urdf_dir, --model_processor`）；
4. 每 task 产出 `eval_result/<suite>/task_<id>/results.json`；云端下改成 `/job_data/eval_result/...`；
5. 支持两个 benchmark：`libero` 与 `libero_plus`；后者额外有 `summarize_libero_plus_categories`（第 245 行起）按 category 汇总。

**运行示例**（本地）：

```bash
export LIBERO_ROOT=$WORKING_PATH/LIBERO
model_config="http://.../checkpoint_50"     # 也可以是本地 exported/model 目录
vlm_ckpt_dir=./ckpt
urdf_dir=./urdf

python3 libero_eval.py \
    --benchmark libero \
    --libero_root $LIBERO_ROOT \
    --model_config $model_config \
    --model_prefix model_0 \
    --vlm_ckpt_dir $vlm_ckpt_dir \
    --urdf_dir $urdf_dir \
    --model_processor libero_processor \
    --task_suite libero_goal \
    --num_trials_per_task 50 \
    --processes_per_gpu 1
```

`--task_suite -1` 会跑全部 4 个 suite。

**云端**：`aidi_submit_config/submit_cfg_libero_eval.json` 与 `submit_cfg_libero_plus_eval.json`。

## 9.4 `robotwin_eval.py`

来源：`projects/holobrain_internal/common/robotwin_eval.py`（157 行）。

**做的事**：
1. 从 `$ROBOTWIN_DIR/envs/*.py` 自动发现任务名；
2. 按 GPU 数轮询分配；
3. 每 GPU 一个 `multiprocessing.Process`，逐个任务 shell 出到 `script/eval_policy.py`（RoboTwin repo 的官方 policy runner）；
4. 用 regex `\d+\.?\d+%` 从 stdout 抓 `Success rate`（第 74-85 行）；
5. 汇总为 `{task: rate}` + 平均 + `num_tasks` + `test_num_per_task`。

**运行示例**：

```bash
export CUDA_VISIBLE_DEVICES=0,1
export ROBOTWIN_DIR=$WORKING_PATH/robotwin
cp -r projects/holobrain_internal/common/holobrain_robotwin_policy $ROBOTWIN_DIR
cp -r projects/holobrain_internal/common/robotwin_eval.py $ROBOTWIN_DIR

cd $ROBOTWIN_DIR
python3 robotwin_eval.py \
    --task_names place_empty_cup,stack_blocks_three \
    --task_config demo_clean \
    --model_config "http://.../checkpoint_50" \
    --vlm_ckpt_dir /path/to/ckpt \
    --urdf_dir /path/to/urdf \
    --test_num 100
```

## 9.5 `robocasa_eval.py`

来源：`projects/holobrain_internal/common/robocasa_eval.py`（926 行）——本仓库里最重的 eval 脚本。

**做的事**：
1. `_bootstrap_robocasa_assets_root` + `prepare_robocasa_runtime` + `register_robocasa_gym_envs`：把 RoboCasa 的 gym env 注册好；
2. `import holobrain_robocasa_policy.{HoloBrainRoboCasaPolicy, ...}`——**在同一进程内构造 policy**；
3. `allocate_tasks_to_gpus`（按 `get_task_horizon` 加权）分片；
4. `evaluate_task_group`：设 `CUDA_VISIBLE_DEVICES`，实例化 policy 一次，然后逐 task 调 `evaluate_task → _evaluate_task → run_trial`；
5. `run_trial` 是一个 deque 循环：`policy.get_action_dicts(obs, env=env)` → 送进 `env.step(action)` 20 Hz；每一步用 `imageio` 录 mp4；
6. 结果写 per-task `log.txt` + JSON summary。

**为什么不用 subprocess**：RoboCasa policy 的模型初始化重（要挂 VLM）；进程内复用同一份 policy 显著更快。

## 9.6 `isaac_eval.py`

来源：`projects/holobrain_internal/common/isaac_eval.py`（295 行）。

**做的事**：
1. 加载 YAML `--multi_task_config`（如 `isaac_task_config/multi_task_setting.yaml`）——含 `_default` 块 + 每 task 的 override，走 `deep_merge_dict` 合并（第 156-176 行）；
2. 对每 task 执行两个 shell 阶段：
   - Phase A：`gen_dualarm_piper_<task_name>.py` 生成 task-specific config；
   - Phase B：`examples/manipulation-app/pick_place/scripts/eval_policy_sem.py` 执行 rollouts，传 `--model_config, --model_processor, --model_prefix, --seed, --rollouts, --maximum_step 1000`；
3. 需要提前 `Xvfb :$id -screen 0 1920x1200x24 -ac +extension GLX +render -noreset &` 起 headless display；
4. 结果表用 `log_task_table`（`AsciiTable`）打印 success / progress。

**运行示例**：见 [common/README.md](../common/README.md) 的 "Isaac Envs" 章节。

**云端**：`aidi_submit_config/submit_cfg_isaac_eval.json`（`docker.hobot.cc/.../isaac_lab-v2.0.2-sem-ext-v0.2`）。

## 9.7 `behavior1k_eval.py`

来源：`projects/holobrain_internal/common/behavior1k_eval.py`（427 行）。

**做的事**：
1. **多节点分布式**：读 `WORLD_SIZE / RANK` 环境变量，走 `torch.distributed.init_process_group(backend="gloo")`；
2. `shard_jobs_by_world` 把任务切分到每个 rank；
3. 每 rank 内起 `Process` × `processes_per_gpu`，用 `multiprocessing.JoinableQueue` 派活；
4. 每个 worker 用 bash template shell 出到 `OmniGibson/omnigibson/learning/eval_b1k.py policy=local +vlm_ckpt_dir=... +urdf_dir=... +model_path=... +model_processor=... +instances_to_run=[...]`；
5. `filelock` 保护模型 URL 下载；
6. 汇总 `q_score.final`（`cal_q_score` 期望 50 tasks）。

**云端**：`aidi_submit_config/submit_cfg_behavior1k_eval.json`——注意 `bash_command_template` 里挂 `Xvfb` + `omnigibson` conda env。

## 9.8 `realworld_eval.py`：Flask 推理服务

来源：`projects/holobrain_internal/common/realworld_eval.py`（597 行）。

**做的事**：
1. `ModelMixin.load_model(model_dir, load_impl="native")` + `HoloBrainProcessor.load(...)` 恢复模型；
2. 可选 `RTCInferencePlugin`（`robo_orchard_lab.models.rtc_plugin.rtc_plugin`）绑定到 `model.decoder.async_inference_plugin`——用于把这一次预测的 `pred_actions` 与上次预测的 `remaining_actions` 做实时融合；
3. Flask + gevent WSGIServer 起服务，`POST /<server_name>`（默认 `/holobrain`，port 6050）接收 multipart 请求：
   - 必需字段：`{left,middle,right}_{color,depth,intrinsic}`, `left_arm_state`, `right_arm_state`, `instruction`；
   - 可选：`remaining_actions`, `delay_horizon`；
4. 内部构造 `MultiArmManipulationInput`（**当前只支持双臂真机**）；
5. Pipeline：`processor.pre_process → model → processor.post_process → _clip_actions_if_needed → interpolate → _limit_action_delta`；
6. 返回 JSON `{"left_arm_actions": [...], "right_arm_actions": [...], "action_horizon": N}`。

**关键运行时旋钮**：

| 参数 | 默认 | 作用 |
|------|------|------|
| `--interpolation` | `200 / 30` | 把模型 30 Hz 输出线性插值到 200 Hz 送给机器人 |
| `--clip_action_len` | 无 | 只输出前 N 步动作 |
| `--delay_horizon` | 无 | RTC 里的延迟规划步数 |
| `--max_action_delta` | 2.0 | 单步动作最大变化幅度；例如 3 会把 `[1, 10]` 拆成 `[1, 4, 7, 10]` |
| `--num_joints_per_arm` | 7 | 单臂关节数 |

**运行示例**：

```bash
python3 realworld_eval.py \
    --port 6050 --server_name holobrain \
    --model_dir ./model --model_url "http://.../checkpoint" \
    --model_processor horizon_beijing_processor \
    --vlm_ckpt_dir /path/to/ckpt --urdf_dir /path/to/urdf \
    --num_joints_per_arm 7 --max_action_delta 2.0
```

## 9.9 `robochallenge_eval.py`

来源：`projects/holobrain_internal/common/robochallenge_eval.py`（132 行）。

**做的事**：
1. 需要 `ROBOCHALLENGE_INFERENCE_REPO` 环境变量指向克隆的 `RoboChallengeInference` repo；
2. 实例化 `holobrain_robochallenge_policy.HoloBrainPolicy(...)`；
3. 两种模式：
   - `--mock`：`InterfaceClient("test_user", mock=True)` + `run_local_client_loop`；
   - 真提交：`--user_token, --submission_id` → `job_loop(client, policy, submission_id, ...)`；
4. `action_type` 根据 embodiment 决定："ur5" / "arx5" 用 `"leftjoint"`，其他用 `"joint"`。

## 9.10 `geniesim3_inference_server.py`

来源：`projects/holobrain_internal/common/geniesim3_inference_server.py`（313 行）。

**做的事**：
1. `build_policy_from_deploy_config` 从 exported model 目录构造 `HoloBrainGenieSim3Policy`；
2. asyncio WebSocket 服务器（`HoloBrainGenieSim3WebsocketServer`）：
   - 连接握手时发 `{"server", "action_dim", "valid_action_step"}`；
   - 循环收 `msgpack.unpackb` 请求 → `policy.get_actions(payload)` → `msgpack.packb` 响应 `{"actions": np.ndarray, "model": "holobrain_geniesim3", "request_count", "error"}`；
   - 出错时返回 shape `(valid_action_step, GENIESIM_ACTION_DIM)` 的零 buffer；
3. CLI：`--model_dir, --model_processor, --model_prefix, --load_impl native|accelerate, --host, --port, --valid_action_step, --sampling_ratio, --gripper_limit, --use_depth`。

**云端**：`aidi_submit_config/submit_cfg_geniesim3_eval.json` 里用 `nohup python3 geniesim3_inference_server.py --port $((8999 + worker_rank)) --use_depth true &` 起服务，`wait_holobrain_start_infer.sh` 阻塞到端口 up，再启动 GenieSim3 仿真。

## 9.11 AIDI 提交配置对照

| 文件 | 用途 | 关键 image 或 script |
|------|------|---------------------|
| `submit_cfg.json` | 主训练 | accelerate + `train.py` |
| `submit_cfg_value_model.json` | value 模型训练 | `config_holobrain_value_common.py` |
| `submit_cfg_libero_eval.json` | LIBERO eval | libero_plus docker |
| `submit_cfg_libero_plus_eval.json` | LIBERO-Plus eval | 同上 + `--benchmark libero_plus` |
| `submit_cfg_robotwin_eval.json` | RoboTwin eval | robotwin2 docker |
| `submit_cfg_robocasa_eval.json` | RoboCasa eval | robocasa docker |
| `submit_cfg_isaac_eval.json` | Isaac eval | isaac_lab-v2.0.2-sem-ext docker |
| `submit_cfg_behavior1k_eval.json` | Behavior-1K eval | omnigibson docker，多节点 gloo |
| `submit_cfg_geniesim3_eval.json` | GenieSim3 eval | 先起 WS server 后启动 sim |

提交命令：

```bash
RoboOrchardJob-AIDISubmit submit_from_config \
    --config projects/holobrain_internal/common/aidi_submit_config/submit_cfg_libero_eval.json
```

## 9.12 快速选路：我该用哪个 eval？

- **想验证 checkpoint 在合成 benchmark 上是否收敛** → `libero_eval` / `robotwin_eval` / `robocasa_eval`。
- **想跑 CVPR RoboChallenge 提交** → `robochallenge_eval` + `submit_cfg_..._eval.json`（若有）。
- **想上机器人** → `realworld_eval` 起 Flask，客户端 HTTP 请求。
- **想接 GenieSim3 官方 harness** → `geniesim3_inference_server`。
- **只需要模型权重的独立包** → `export.py`。

---

**下一篇 →** [10_logging_and_debug.md](./10_logging_and_debug.md)
