# 03 — 评测侧完整通路（外部 RoboDojo repo 流程）

> ## ⚠️ 适用性：本文描述的**编排层已被取代**
>
> 现在跑评测**不用**本文 §1–§2 描述的外部 RoboDojo repo 流程，改用同事 xuewu.lin 的
> **in-repo 评测器** `projects/holobrain_internal/common/robodojo_eval.py`
> （配 `holobrain_robodojo_policy/` + 官方镜像 `robotlab-mani:...robodojo-v0.5`）。
> 提交配置在 `common/aidi_submit_config/submit_cfg_robodojo_eval_kun_*.json`，
> 命令见 [04_commands_cheatsheet.md](04_commands_cheatsheet.md) §1.2。
>
> **本文哪些还有效**：
> - §3 Policy server 端、§4 Env client 端、§5 Obs/Action wire 格式、§6 Episode loop、
>   §7 结果输出 schema、§8 中止/中断 —— **两套流程共用**，这些仍然准确，是理解
>   RoboDojo 内部机制最详细的一份记录。
> - §1 AIDI 提交端、§2 评测入口链（`scripts/robodojo.sh` / `smoke_all_tasks.sh`）
>   —— **仅适用于旧流程**，作为历史记录保留。
>
> **为什么换**：并发/GPU 利用率、进程隔离、官方镜像、以及内置官方 protocol 汇总
> （`_write_benchmark_summary`）全面更优。旧流程的 seed0 job 只跑到 13/54 run-config
> 就被主动停掉（2 卡跑 8 卡分配，浪费 6 卡）。
>
> **最新结果**：见 [07_results.md](07_results.md)。

从 AIDI submit 到 `_result.json` / `episode_*.mp4` 落 bucket 的完整链路。
本文语境下的核心 repo 是 `~/git_repo/RoboDojo/`（现行流程则把评测代码放在
robo_orchard_lab 的 `projects/holobrain_internal/common/` 下）。

**本文的默认 job**：`bcloud-bj-zone1-7895445e92bc`（旧流程 seed0 eval，**已 Stopped，
仅完成 13/54 run-config**，结果在 `/horizon-bucket/.../robodojo-holobrain-seed0/`）

---

## 1. AIDI 提交端

### 1.1 submit_cfg_holobrain_robodojo_seed0.json

**文件**：`~/git_repo/RoboDojo/aidi_submit/cfgs/submit_cfg_holobrain_robodojo_seed0.json`

| 字段 | 值 | 作用 |
|---|---|---|
| `job_name` | `kun01wu_robodojo_holobrain_seed0` | AIDI job name |
| `workspace_folder` | `aidi_workspace_holobrain_seed0_kun` | AIDI 拷贝 `to_upload` 到该目录再 rsync 到 pod |
| `clear_workspace` | `true` | 否则多次提交会 pile up 到几十 GB |
| `docker_image` | `docker.hobot.cc/imagesys/kun01.wu/robodojo-holobrain:ubuntu22.04-gcc11.4-cu128-torch280-holobrain-20260727-v6` | 私有镜像（含 IsaacSim + `holobrain` / `RoboDojo` 双 env） |
| `input_bucket` | `robot_lab,robot_lab2` | 两个 bucket 只读挂载 |
| `output_bucket` | `robot_lab` | 结果写 `robot_lab` |
| `num_workers` | `1` | 单节点 |
| `gpu_per_worker` | `8` | 8 卡（policy 用 0，env 用 1，其它闲置） |
| `wall_time` | `2880` | 分钟 = 48h，超时 SIGTERM |
| `queue_name` | `project-5090-robot-lab-bcloud-bj` | 5090 集群 |
| `project_id` | `horizon-labs` | 计费 |
| `to_upload` | `["env","env_cfg","task","src","utils","scripts","XPolicyLab","docs","aidi_submit","pyproject.toml","robo_orchard_lab"]` | 打包 → `/running_package/code_package/` |

**注意**：`to_upload` 里的 `robo_orchard_lab` 是 RoboDojo 侧的 wrapper 目录（`cp -r ~/git_repo/robo_orchard_lab/robo_orchard_lab RoboDojo/robo_orchard_lab/robo_orchard_lab`），**不能是 symlink**——AIDI 的 `rsync -aL` 实测会 dangling 到 dev machine 路径。见 [05_troubleshooting.md](05_troubleshooting.md) §「rsync -aL 坑」。

### 1.2 cmd 关键段（逐块解释）

```bash
# L1-8: 环境初始化
set -euo pipefail
source /opt/miniconda3/etc/profile.d/conda.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
export TMPDIR=/tmp/isaaclab_kun

# L9: 让 IsaacSim 能找到 conda env 里的 libpython
export LD_LIBRARY_PATH=/home/users/kun01.wu-labs/miniconda3/envs/RoboDojo/lib:$LD_LIBRARY_PATH

# L10-11: HuggingFace proxy 打到无效地址，防止 IsaacLab boot 时反向代理去下模型
export DEPLOY_PROXY_HOST=127.0.0.1 DEPLOY_PROXY_PORT=1

# L12: 让 policy server 能 import robo_orchard_lab
export PYTHONPATH=${WORKING_PATH}:${WORKING_PATH}/robo_orchard_lab:$PYTHONPATH

# L13-16: 4 个必需 symlink
ln -sfn /horizon-bucket/robot_lab/users/kun01.wu/datasets/RoboDojo/Assets  Assets
ln -sfn /horizon-bucket/robot_lab2/datasets/all_data/urdf/urdf_v20260711   urdf
ln -sfn /horizon-bucket/robot_lab/users/xuewu.lin/ckpt                     ckpt
ln -sfn /horizon-bucket/robot_lab/users/kun01.wu/robo_orchard_lab/ckpts/holobrain_robodojo_posttrain_v9/checkpoint_20000 \
        XPolicyLab/policy/HoloBrain/checkpoints/checkpoint_20000

# L17-18: 改写 env_cfg 里的绝对路径为 pod 侧的实际路径
conda run -n RoboDojo python utils/update_embodiment_config_path.py

# L19-20: IsaacLab pin sed patch（关键！）
sed -i 's|isaacsim.asset.importer.urdf" = {version = "2.4.31", exact = true}|isaacsim.asset.importer.urdf" = {}|' \
    /home/users/kun01.wu-labs/git_repo/RoboDojo/third_party/IsaacLab/apps/isaaclab.python.kit \
    ${WORKING_PATH}/third_party/IsaacLab/apps/isaaclab.python.kit 2>/dev/null || true
```

**为什么 L19-20 必须有**：docker image 里 `isaacsim.asset.importer.urdf` 是 2.4.30；`IsaacLab` 硬 pin 到 `2.4.31 exact`，Kit boot 时找不到 exact match 会立即 exit。把 pin 放开成任意版本才能启动。两条路径都 sed 是因为 image 里和 workspace 里各有一份 kit 文件。

```bash
# L21-24: bucket-side OUT_DIR & symlink
OUT_DIR=/horizon-bucket/robot_lab/users/kun01.wu/robo_orchard_lab/eval_results/robodojo-holobrain-seed0
mkdir -p $OUT_DIR/{eval_result,smoke_results}
export ROBODOJO_EVAL_ROOT=$OUT_DIR/eval_result       # scripts/internal/summarize_result.py:37 读
ln -sfn $OUT_DIR/eval_result    eval_result
ln -sfn $OUT_DIR/smoke_results  smoke_results

conda activate RoboDojo   # sim env（policy server 内部会切到 holobrain）

# L26-32: 背景 rsync 保险丝（每 60s + trap EXIT flush）
(while true; do
   rsync -a smoke_results/ $OUT_DIR/smoke_results/ 2>/dev/null || true
   rsync -a eval_result/   $OUT_DIR/eval_result/   2>/dev/null || true
   sleep 60
 done) &
BG_RSYNC_PID=$!
trap 'kill $BG_RSYNC_PID 2>/dev/null || true;
      rsync -a smoke_results/ $OUT_DIR/smoke_results/ 2>/dev/null || true;
      rsync -a eval_result/   $OUT_DIR/eval_result/   2>/dev/null || true' EXIT
```

**保险丝的意义**：`eval_result/` 和 `smoke_results/` 已经是 symlink 到 bucket，写入本已直落 bucket。但 IsaacSim / ffmpeg 可能开 buffer，最终 flush 要 `trap EXIT` 保证。

```bash
# L33: 主评测命令
bash scripts/robodojo.sh benchmark \
    --policy-dir XPolicyLab/policy/HoloBrain \
    --ckpt checkpoint_20000 \
    --env-cfg arx_x5_holobrain \
    --action-type joint \
    --seed 0 \
    --policy-gpu 0 --env-gpu 1 \
    --policy-env holobrain --eval-env RoboDojo \
    --eval-num 25 2>&1 | tee $OUT_DIR/benchmark.log

# L34: 结果汇总 markdown
python scripts/internal/summarize_result.py 2>&1 | tee $OUT_DIR/summary.log
```

- **`--eval-num 25`**：每 task 跑 25 episode（xiaomi 是 100，为省时间减到 25）
- **`--fail-fast` 未启用**：一个 task 挂了不影响后面 task
- **`--action-type joint`**：14 维 joint control（left 6 arm + 1 gripper, right 6 arm + 1 gripper）

### 1.3 提交命令

```bash
cd /home/users/kun01.wu-labs/git_repo/RoboDojo
RoboOrchardJob-AIDISubmit submit_from_config \
    --config aidi_submit/cfgs/submit_cfg_holobrain_robodojo_seed0.json
```

（会打印 `Command executed:` 后即视为已提交，job_id 通过 job/list REST API 反查——见 [04_commands_cheatsheet.md](04_commands_cheatsheet.md) §「查 job_id」）。

### 1.4 arx_x5_holobrain env_cfg

env_cfg 名字 `arx_x5_holobrain` 通过 `./env_cfg/<name>.yml` 解析：

- **top-level**：`env_cfg/arx_x5_holobrain.yml`
  - `observation.vision.intrinsic_matrix: true` + `extrinsic_matrix: true`（HoloBrain 独有，比 base `arx_x5.yml` 多出这两项）
- **sub-configs 引用链**：
  - `env_cfg/sim/sim_config.yml`
  - `env_cfg/scene/default.yml`
  - `env_cfg/camera/camera_config.yml` — 定义 3 cam：`cam_head`, `cam_left_wrist`, `cam_right_wrist`
  - `env_cfg/robot/dual_x5.yml` → 每 arm USD 在 `env/robot_manager/robot_config/x5.py:11`：`usd_path=f"{ROBOTS_PATH}/x5/ARX.usd"`（就是 log 里 `X5A.usd` 引用的那个）

- **task registry**（每个 task 的 python class）：`task.RoboDojo.task_registry.load_task_class(task_name)`，被 `src/eval_client/main.py:70` + `src/eval_client/eval_env.py:48-49` 调用。

---

## 2. 评测入口链

### 2.1 `scripts/robodojo.sh benchmark`

**文件**：`~/git_repo/RoboDojo/scripts/robodojo.sh`

- Line 482：`benchmark` case dispatch
- Line 435-460：`run_sweep benchmark` → `bash scripts/internal/smoke_all_tasks.sh --eval-num $EVAL_NUM ...`

### 2.2 `scripts/internal/smoke_all_tasks.sh`

**文件**：`~/git_repo/RoboDojo/scripts/internal/smoke_all_tasks.sh`

主循环骨架：

```bash
# smoke_all_tasks.sh:181-188 — 拿 runnable task list
mapfile -t TASKS < <(python scripts/internal/task_inventory.py --only-runnable)   # → 54 tasks

# smoke_all_tasks.sh:305-381 — 逐 task 调 eval
for task in "${TASKS[@]}"; do
    echo "[smoke_all_tasks] RUN $task"
    log_dir="${ROOT_DIR}/smoke_results/${run_id}/logs"
    bash scripts/robodojo.sh eval \
        --dataset RoboDojo --task $task --ckpt checkpoint_20000 \
        --env-cfg arx_x5_holobrain --expert-num 100 --action-type joint \
        --seed 0 --policy-gpu 0 --env-gpu 1 \
        --policy-env holobrain --eval-env RoboDojo \
        --policy-dir XPolicyLab/policy/HoloBrain \
        --eval-num 25 > "${log_dir}/${task}.log" 2>&1
    record_result $task $? ...       # L278-294
    write_summaries                  # L229-276 → smoke_results/${run_id}.{json,md}
done
```

### 2.3 `<run_id>.json` schema

```json
{
  "run_id": "2026-07-27_21-49-05_smoke",
  "eval_num": 25,
  "dimensions": ["all"],
  "counts": {"PASS": 5, "FAIL": 0, "SKIP": 0, "DRY_RUN": 0},
  "results": [
    {"status": "PASS", "task": "align_blocks", "exit_code": "0",
     "eval_time": "25", "elapsed_sec": "1809",
     "result_path": ".../\_result.json", "log_path": ".../align_blocks.log",
     "message": "ok"},
    ...
  ]
}
```

- **PASS 定义**（`smoke_all_tasks.sh:369`）：`rc==0 && eval_time>=1`

### 2.4 `scripts/robodojo.sh eval` → `run_policy_eval.sh`

**文件**：`~/git_repo/RoboDojo/scripts/internal/run_policy_eval.sh`

```bash
# run_policy_eval.sh:56 — 分配空闲端口
PORT=$(bash XPolicyLab/utils/get_free_port.sh)

# run_policy_eval.sh:60-90 — trap INT/TERM/EXIT → 干掉 server 进程树
trap cleanup EXIT INT TERM

# run_policy_eval.sh:94-108 — bg 起 policy server
bash XPolicyLab/policy/HoloBrain/setup_eval_policy_server.sh \
    $bench_name $task $ckpt $env_cfg $action_type $seed $policy_gpu $policy_env $PORT &
SERVER_PID=$!

# run_policy_eval.sh:110-115 — 等 server 就绪（timeout 600s）
bash XPolicyLab/utils/wait_for_policy_server.sh $PORT 600

# run_policy_eval.sh:117-130 — 起 env client
bash XPolicyLab/policy/HoloBrain/setup_eval_env_client.sh ...

echo "[MAIN] eval finished"                # L132
kill $SERVER_PID; wait
```

---

## 3. Policy server 端

### 3.1 setup_eval_policy_server.sh

**文件**：`~/git_repo/RoboDojo/XPolicyLab/policy/HoloBrain/setup_eval_policy_server.sh`

```bash
# L25:  deploy.yml = ${policy_dir}/deploy.yml
# L41-42: conda activate holobrain    ← 关键：切到 holobrain env
# L45-47: export PYTHONPATH="${BENCH_ROOT}/robo_orchard_lab:${PYTHONPATH}"  + HF_HUB_OFFLINE=1
# L49-64:
exec env CUDA_VISIBLE_DEVICES=$policy_gpu_id python XPolicyLab/setup_policy_server.py \
     --config_path deploy.yml \
     --overrides port=... host=... \
                 bench_name=... task_name=... ckpt_name=checkpoint_20000 \
                 env_cfg_type=arx_x5_holobrain seed=0 policy_name=HoloBrain \
                 action_type=joint action_dim=...
```

### 3.2 setup_policy_server.py 主流程

**文件**：`~/git_repo/RoboDojo/XPolicyLab/setup_policy_server.py`

```python
# setup_policy_server.py:22-71 main()
model_class_func = eval_function_decorator(f"XPolicyLab.policy.{policy_name}.model", "Model")   # L31
model = model_class_func(deploy_cfg)                                                             # L32
# ws 分支：
from client_server.ws.model_server import PolicyServer, PolicyServerConfig
server = PolicyServer(model, PolicyServerConfig(host=host, port=int(port)))
asyncio.run(server.serve_forever())                                                              # L68
```

### 3.3 HoloBrain Model.__init__ — checkpoint 加载

**文件**：`~/git_repo/RoboDojo/XPolicyLab/policy/HoloBrain/model.py:315-387`

```python
model_dir = _resolve_ckpt_dir(model_cfg)     # → checkpoints/checkpoint_20000    (L336)
_ensure_urdf_visible(model_dir)              # L364: 若 urdf/ 缺失就 symlink 一份
_patch_holobrain_vlm_attn_to_sdpa()          # L355: rewrite Qwen*VL flash_attn → sdpa
                                              #        (v6 image 里 flash_attn 未装)

from robo_orchard_lab.models.holobrain.processor import HoloBrainProcessor
from robo_orchard_lab.models.mixin           import ModelMixin

self.processor = HoloBrainProcessor.load(
    str(model_dir), f"{self.processor_name}.json")           # L369 → robodojo_processor.json
self.model = ModelMixin.load_model(
    str(model_dir), model_prefix="model", load_impl="native") # L372 → model.safetensors + model.config.json

self.model.eval()
self.model.requires_grad_(False)
self.model.to(self.device)                                    # L375-377
```

- **`self.processor_name`** 从 `deploy.yml:` 里读，值是 `"robodojo"` → 于是读 `robodojo_processor.json`
- 部署时 `HoloBrainProcessor.load` 会 `os.chdir(model_dir)` 之类，因此需要 `./urdf/...` + `./ckpt/Qwen2.5-VL-3B-Instruct` 都可解析 —— 见 [02_deploy_package.md](02_deploy_package.md)

### 3.4 Protocol / codec

**文件**：`~/git_repo/RoboDojo/XPolicyLab/client_server/ws/protocol/codec.py`

- `encode_frame(Frame|dict) -> bytes` (L40): `msgpack.packb(..., default=_encode_numpy, use_bin_type=True)`
- `_encode_numpy` (L16): 把 numpy array 序列化为 `{ndim, dtype, shape, raw bytes}`（走 `msgpack_numpy`）
- `decode_frame(bytes) -> dict` / `decode_envelope(bytes) -> Frame` (L53-71): 反向

**Frame schema** (`XPolicyLab/client_server/ws/protocol/schemas.py:31`):
```python
class Frame:
    message_type: MessageType   # HELLO/HELLO_ACK/RESET/RESET_RESULT/INFER/INFER_RESULT/CLOSE
    request_id: str
    evaluation_id: str
    action_case_id: str
    trial_id: str
    repeat_index: int
    step: int
    sent_at: float
    payload: dict
```

### 3.5 server handlers

**文件**：`~/git_repo/RoboDojo/XPolicyLab/client_server/ws/model_server.py`

- `_handle_reset` (L212)：`model.reset()` 加锁后调用，返回 `RESET_RESULT`
- `_handle_infer` (L222):
  ```python
  observation = frame.payload.get("observation")     # L223
  update_obs  = getattr(self.model, "update_obs", None)
  get_action  = getattr(self.model, "get_action", None)
  # 线程内锁执行 update_obs(observation) → result = get_action()
  payload = {"actions": result, "latency_ms": latency_ms}     # L272
  return self._reply(frame, MessageType.INFER_RESULT, payload)
  ```

---

## 4. Env client 端

### 4.1 客户端启动链

```
setup_eval_env_client.sh → XPolicyLab/utils/setup_env_client.sh →
XPolicyLab/utils/run_sim_env_client.sh:26 → scripts/eval_policy.sh:146
   ↓
python -u src/eval_client/main.py \
    --task_name $task --env_cfg_type arx_x5_holobrain \
    --num_envs 1 --enable_cameras --kit_args "..." \
    --device_id 1 --policy_name HoloBrain \
    --port $PORT --protocol ws --policy_server_url ws://localhost:$PORT \
    --additional_info "ckpt_name=checkpoint_20000,action_type=joint" \
    --seed 0 --host localhost --headless
```

### 4.2 main.py 主流程

**文件**：`~/git_repo/RoboDojo/src/eval_client/main.py`

```python
# main.py:12-64  argparse
# main.py:124-125  AppLauncher(args_cli) → 启动 IsaacSim (headless)
# main.py:249-444  main()
def main():
    # L277-305: OmegaConf merge YAML: sim/scene/camera/robot/task_env/eval_cfg/deploy_cfg
    env_cfg = compose_env_cfg(args_cli)
    # L341: 建评测环境（含 policy client）
    env = create_eval_env(env_cfg, simulation_app, resume_state=resume_state)

    # L348-437: 主循环
    while env.env_seeds is not None:
        env.reset(seed=env.env_seeds)   # spawn scene + init robot + WsModelClient.reset
        env.run_eval()                  # 一批 episode 端到端
        env.seed_manager.eval_step()
        # 从 seed_manager 拿下一批种子...

    print(f"Success nums: {env.success_nums}, "
          f"Fail nums: {env.fail_nums}, Unstable nums: {env.unstable_nums}")   # L429
    _delete_resume_manifest(env)        # L441
```

### 4.3 create_eval_env — task 类动态子类化

**文件**：`~/git_repo/RoboDojo/src/eval_client/eval_env.py:43-...`

`create_eval_env` 动态继承 `task.RoboDojo.task_registry.load_task_class(task_name)`（即每个 task 的 py 类），把 EvalEnv 的通用逻辑和 task-specific 逻辑合到一起。

内部会实例化 4 大 manager（依赖 `env_cfg`）：

| Manager | 作用 | 文件 |
|---|---|---|
| `robot_manager` | 加载 dual x5 arm + articulation | `env/robot_manager/robot_manager.py`（配置 `robot_config/x5.py:8-11`：`ARX.usd`） |
| `scene_manager` | 场景元素、layout 随机 | `env/scene_manager/*` |
| `camera_manager` | 3 cam 采集，intrinsic/extrinsic 生成 | `env/camera_manager/camera_manager.py:454-489` |
| `reward_manager` | success/fail 判定 | `env/reward_manager/reward_manager.py:527` |

### 4.4 env.reset / setup_scene

```python
# eval_env.py:206-243  reset()
super().reset()                           # Isaac Sim step, load stage
self.setup_scene()                        # L245-262: 200 sim-step warm-up + check_layout_stability
self.model_client.call(func_name="reset")  # L243: 打 RESET frame 到 policy server → model.reset()
```

---

## 5. Obs / Action dict on the wire

### 5.1 Obs dict（client → server, in INFER payload）

**构造位置**：`env/observation_manager/obs_manager.py:76-201`
**消费位置**：`XPolicyLab/policy/HoloBrain/model.py:392-425`

```python
obs = {
    "data_format_version": "v1.0",
    "additional_info": {"frequency": 25},                # L87
    "instruction": "<task description string>",          # L89
    "env_idx": 0,                                         # by EvalEnv.get_obs_batch:281
    "vision": {
        "cam_head": {
            "color":            ndarray(H, W, 3) uint8 RGB,        # L101, RGB slice [:,:,:3] at L104
            "shape":            (H, W, 3),                         # L108, H×W 由 Gemini_345Lg 决定
            "intrinsic_matrix": ndarray(3, 3) float64,             # L136 (camera_manager.py:454-467)
            "extrinsic_matrix": ndarray(4, 4) float64,             # L140 (camera_manager.py:469-489, cam→world USD 惯例)
        },
        "cam_left_wrist":  {同上},
        "cam_right_wrist": {同上},
    },
    "state": {
        "left_arm_joint_state":  ndarray(6,) float,       # L154, robot.arm_name="left_arm"
        "right_arm_joint_state": ndarray(6,) float,
        "left_ee_pose":          ndarray(7,) float,       # [x, y, z, qw, qx, qy, qz]
        "right_ee_pose":         ndarray(7,) float,
        "left_ee_joint_state":   ndarray(1,) float,       # L186-188 (gripper 0..1 normalized)
        "right_ee_joint_state":  ndarray(1,) float,
    },
    "action": {  # 上一步 action echo，HoloBrain 不使用
        ...
    },
}
```

### 5.2 Action list（server → client, in INFER_RESULT payload）

**构造位置**：`XPolicyLab/policy/HoloBrain/model.py:430-466`

`Model.get_action()` 返回 `list[dict]` 长度 = `chunk_size=4`（`deploy.yml:18`）。每个 dict：

```python
{
    "left_arm_joint_state":  ndarray(6,) float32,     # 6 joint of left arm
    "left_ee_joint_state":   ndarray(1,) float32,     # 1 gripper
    "right_arm_joint_state": ndarray(6,) float32,
    "right_ee_joint_state":  ndarray(1,) float32,
}
```

- 内部 raw output 是 `[14]` = `[left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]`，split 到 4 个 key（L430-448）
- **shape check**：`EvalEnv.validate_action_dict` (`eval_env.py:560-634`) 根据 `self.robot_action_dim_info` 强制每 key 的 dim

### 5.3 Client wire 顺序

**文件**：`~/git_repo/RoboDojo/XPolicyLab/client_server/ws/model_client.py:34-`

```python
def call(self, func_name, obs=None, ...):
    if func_name == "reset":
        return self.client.reset(...)                    # RESET frame (L46)
    if func_name == "update_obs":
        self._latest_obs = obs                           # L60 — 只缓存，不发！
        return None
    if func_name == "get_action":
        resp = self.client.infer(self._latest_obs)       # L64 → INFER frame with payload={"observation": obs}
        return resp.payload["actions"]                    # list[dict] chunk of 4
```

**关键**：`update_obs` **不发网络**，只本地缓存。每个 policy step 只有 1× INFER wire。

---

## 6. Episode loop

**文件**：`~/git_repo/RoboDojo/XPolicyLab/policy/HoloBrain/deploy.py:17-32`

```python
def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")                              # L17-18
    while not TASK_ENV.is_episode_end():                              # L20
        obs = TASK_ENV.get_obs()                                       # L21
        model_client.call(func_name="update_obs", obs=obs)             # L22
        actions = model_client.call(func_name="get_action")            # L23 ← chunk of 4 dicts
        for action_idx, action in enumerate(actions):
            TASK_ENV.take_action(action)                               # L26
            if TASK_ENV.is_episode_end() or action_idx + 1 == len(actions):
                break
            obs = TASK_ENV.get_obs()                                   # L31
            model_client.call(func_name="update_obs", obs=obs)         # L32 (mid-chunk resync)
```

### 6.1 Obs 采集细节

`TASK_ENV.get_obs()` → `EvalEnv.get_obs()` (`eval_env.py:264`) → `get_obs_batch([0])[0]` (`eval_env.py:265`) → `ObsManager.get_obs([0])` (`obs_manager.py:76`):

```python
# obs_manager.py 关键
self.capture_manager.step(env_ids=[0])        # L92 — 触发 3 cam 抓取
obs[0]["vision"]["cam_head"]["color"] = camera_data[:, :, :3]   # L101-104 RGB
obs[0]["state"]["left_arm_joint_state"]  = robot_manager.get_joint(...)     # L151
obs[0]["state"]["left_ee_pose"]          = robot_manager.get_real_endpose(...)  # L158
```

### 6.2 Video stream 副产品

`EvalEnv._stream_vision(env_idx, frame)` (`eval_env.py:870-893`) 每帧调 `VideoStreamWriter` 写 `${save_dir}/_stream/env0_cam_head.tmp.mp4`。

### 6.3 Action apply

```python
# eval_env.py:352-354  take_action(action)
self.take_action_batch([action], env_idx_list=[0])

# eval_env.py:375-404  action_type=joint 分支
control_info = build_arm_joint_control(action)  # gripper 0..1 → robot.gripper_scale
# eval_env.py:473-558  process_control_info: linear interp 首 80% + hold 20%
# eval_env.py:448-453  robot_manager.control_manager.push + sim step 直到队列空
```

### 6.4 Success/fail 判定

**文件**：`~/git_repo/RoboDojo/src/eval_client/eval_env.py:839-865`

```python
def is_episode_end(self, final_check=False):
    reward_list = self.reward_manager.get_reward(final_check=final_check)   # L848
    for env_idx in range(self.num_envs):
        if self.end_flag[env_idx]: continue
        if reward_list[env_idx] > 1 - 1e-3:                                  # L852: reward==1 → success
            self.end_flag[env_idx] = True
            self.success[env_idx]  = True
            continue
        if self.take_action_cnt[env_idx] >= self.step_lim or not self.success[env_idx]:
            self.end_flag[env_idx] = True
            self.success[env_idx]  = False                                   # L858: step_lim → fail
```

- **`reward_list[env_idx] > 1 - 1e-3`** → 该 task 的所有 check/query/trigger list 都被清空
- **`step_lim`** 是每 task 自己定义的（如 `task/RoboDojo/tasks/arrange_largest_number.py:11`: `self.step_lim = 1050`）
- **reward 源**：`RewardManager.get_reward` (`env/reward_manager/reward_manager.py:527-587`) — 只有 task-specific 检查函数全 pass 才 return 1.0

**为什么 log 里 `Success nums: 0, Fail nums: 25`**：policy 从来没让 `reward > 1-1e-3`，每个 layout 都在 `step_lim` 步内没完成 → 全 fail。

---

## 7. 结果输出

### 7.1 `_result.json`

**写位置**：`EvalEnv.run_eval()` (`eval_env.py:830`):

```python
save_json(self.eval_result, os.path.join(self.save_dir, "_result.json"))
```

**save_dir 路径**（`eval_env.py:80-88`）:

```
eval_result/RoboDojo/<task>/HoloBrain/arx_x5_holobrain/<seed>_ckpt_name=checkpoint_20000,action_type=joint/<ROBODOJO_RUN_ID>/_result.json
```

**JSON 结构**（`eval_env.py:130-135` + `:793-829`）:

```json
{
  "success_rate": 0.0,          // success_nums / eval_time (L827)
  "eval_time": 25,              // = success_nums + fail_nums (L829)
  "score": 0.4,                 // total_score / eval_time * 100 (L828)
  "details": {                  // L811-815
    "0": {"layout_id": 0, "success": false, "score": 0.0},
    "1": {"layout_id": 1, "success": false, "score": 0.05},
    ...
  }
}
```

`save_json` 位于 `utils/save_file.py:134-`（atomic: tmp + rename, `indent=2`）。

### 7.2 视频文件命名

**writer**：`VideoStreamWriter` (`utils/save_file.py:24-131`) — ffmpeg pipe: `rawvideo → libx264 crf=23 yuv420p`
**finalize**：`EvalEnv.save_video(env_idx, video_path, tag)` (`eval_env.py:919-940`):

```python
final_path = video_path.replace(".mp4", f"_{cam_key}_{tag}.mp4")   # L923
writer.close(announce=False)
os.replace(tmp_path, final_path)                                    # L931
```

- `video_path = os.path.join(self.save_dir, f"episode_{index:07d}.mp4")` (L816)
- `tag ∈ {"success", "fail"}` (L796-801)

产物示例：

```
episode_0000000_cam_head_fail.mp4
episode_0000000_cam_left_wrist_fail.mp4
episode_0000000_cam_right_wrist_fail.mp4
```

### 7.3 `smoke_results/<run_id>.json` summary

由 `write_summaries()` (`scripts/internal/smoke_all_tasks.sh:229-276`) 每 task 完写一次。字段见 §2.3。同时写 markdown 表 (L256-274)。

### 7.4 跨 task 汇总 markdown

`scripts/internal/summarize_result.py` (cmd L34 调):
- 读 `$ROBODOJO_EVAL_ROOT` = `$OUT_DIR/eval_result`
- 遍历每个 `_result.json`
- 应用规则「独立 task 50 ep, `<task>_random` + `<task>` 各 25 ep → 拼成 50 ep 一组」(`summarize_result.py:43-45`, `13-19`)
- 写 `$ROBODOJO_EVAL_ROOT/_summary.md`（policy × seed × dimension 分组）

---

## 8. 中止 / 中断

### 8.1 Wall-time / SIGTERM 路径

- AIDI 到 `wall_time=2880` min → SIGTERM 打给顶层 shell（cmd 字符串）
- `trap ... EXIT` (submit_cfg L26-32) 触发：kill bg rsync + 一次 flush rsync
- `run_policy_eval.sh:60-90` 的 `trap cleanup EXIT INT TERM` → `_kill_process_tree` TERM 然后 KILL policy server 进程树
- Partial `_result.json` + partial `.mp4` 都会被 flush 到 bucket

### 8.2 Resume manifest

**文件**：`~/git_repo/RoboDojo/src/eval_client/eval_env.py`

- **manifest 路径** (`resume_manifest_path`, L651-667):
  ```
  eval_result/RoboDojo/<task>/HoloBrain/<config>/<seed>_<additional_info>/_resume_<ROBODOJO_RUN_ID>.json
  ```
- **写者** `persist_resume_manifest` (L669-707): atomic (tmp + `os.replace`)。字段：`run_id, save_dir, task_name, policy_name, config_name, eval_seed, additional_info, success_nums, fail_nums, unstable_nums, total_score, completed_layout_ids, abandoned_layout_ids, details, restart_count`
- **调用点**：
  - 每 batch 结束 `EvalEnv.run_eval()` (L834-837) — best-effort
  - PhysX in-process 恢复 `main._restart_or_exit()` (`main.py:215`) — 在 `os.execv` 前
  - PhysX shell-level `main._exit_for_shell_restart()` (`main.py:238`)
- **读者** `main._load_resume_manifest(eval_cfg, run_id)` (`main.py:160-178`)
- **消费**：`EvalEnv.__init__` (L90-175) 恢复 `save_dir, success_nums, fail_nums, total_score, details, abandoned_seeds`，然后 `seed_manager.init_eval(completed_layout_ids, abandoned_layout_ids)` (L172) 让种子队列跳过已完成 layout
- **完成后清理**：`main._delete_resume_manifest(env)` (`main.py:441`) 删掉 manifest

### 8.3 Bash-level 重试

`scripts/eval_policy.sh:142-185`:
- rc=99 (in-process cap)、134 (SIGABRT / PhysX C++)、139 (SIGSEGV) → sleep 5s + 重 exec `python -u src/eval_client/main.py`
- 最多 `ROBODOJO_MAX_BASH_RETRIES=10` 次
- 用同一个 `ROBODOJO_RUN_ID` 触发 resume

### 8.4 Task 完成打印

- `[MAIN] eval finished` — `run_policy_eval.sh:132`（client 正常 rc=0 后）
- `[robodojo eval] wall_clock=Ns` — `scripts/robodojo.sh:209`（子 shell 返回后）
- `[eval_policy] ROBODOJO_RUN_ID=…` — `scripts/eval_policy.sh:140`

---

## 相关文件汇总

| 目的 | 路径 |
|---|---|
| AIDI submit config | `aidi_submit/cfgs/submit_cfg_holobrain_robodojo_seed0.json` |
| Benchmark dispatcher | `scripts/robodojo.sh` |
| 顺序 task sweep | `scripts/internal/smoke_all_tasks.sh` |
| 单 task server+client 启动 | `scripts/internal/run_policy_eval.sh` |
| Policy server bootstrap (HoloBrain) | `XPolicyLab/policy/HoloBrain/setup_eval_policy_server.sh` |
| Env client bootstrap | `XPolicyLab/policy/HoloBrain/setup_eval_env_client.sh` → `utils/setup_env_client.sh` → `utils/run_sim_env_client.sh` → `scripts/eval_policy.sh` |
| Sim client 入口 (Python) | `src/eval_client/main.py` |
| EvalEnv 主类 | `src/eval_client/eval_env.py` |
| Observation collector | `env/observation_manager/obs_manager.py` |
| Camera intrinsic/extrinsic | `env/camera_manager/camera_manager.py:454, 469` |
| Robot USD spawn | `env/robot_manager/robot_config/x5.py:11` (`ARX.usd`) |
| Task success predicate | `env/reward_manager/reward_manager.py:527` |
| Policy server entry | `XPolicyLab/setup_policy_server.py:22` |
| Policy WS server | `XPolicyLab/client_server/ws/model_server.py:52`(start), `:222`(_handle_infer) |
| Policy WS client (env-side) | `XPolicyLab/client_server/ws/model_client.py:34` (`WsModelClient.call`) |
| Protocol codec | `XPolicyLab/client_server/ws/protocol/codec.py` (msgpack + msgpack_numpy) |
| Frame schema | `XPolicyLab/client_server/ws/protocol/schemas.py:31` |
| HoloBrain policy adapter | `XPolicyLab/policy/HoloBrain/deploy.py`, `model.py` (Model class + ckpt load) |
| HoloBrain deploy config | `XPolicyLab/policy/HoloBrain/deploy.yml` |
| Video writer | `utils/save_file.py:24-131` |
| `_result.json` writer | `src/eval_client/eval_env.py:830` (`run_eval`) |
| `smoke_results/<run_id>.json` writer | `scripts/internal/smoke_all_tasks.sh:229-276` |
| Cross-task 汇总 | `scripts/internal/summarize_result.py` |
| Resume manifest writer | `src/eval_client/eval_env.py:669-707` |
| Resume manifest reader | `src/eval_client/main.py:160-178` |
| Bash-level PhysX retry | `scripts/eval_policy.sh:142-185` |
| In-process PhysX restart | `src/eval_client/main.py:206-246` |
