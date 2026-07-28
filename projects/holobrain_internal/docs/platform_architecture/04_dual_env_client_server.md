# 04 — 双 conda env (Policy Server + Env Client) 机制

**问题**：评测 HoloBrain on RoboDojo 需要同时跑 **policy inference (要 transformers 5.10 + torch 2.8)** 和 **IsaacSim (硬绑 py3.10)**。两者依赖树严重冲突，装不进同一个 env。怎么办？

**答案**：一个 pod、两个 conda env、两个进程、WebSocket 通信、GPU 隔离。以 `~/git_repo/RoboDojo/aidi_submit/cfgs/submit_cfg_holobrain_robodojo_seed0.json` 为例。

---

## 1. 全景图

```
┌─────────────  1 pod × 8 × RTX 5090  ───────────────────────────────────┐
│                                                                          │
│  docker.hobot.cc/imagesys/kun01.wu/robodojo-holobrain:...-v6            │
│  镜像里预置两个 conda env：                                              │
│     /opt/miniconda3/envs/holobrain      (py3.11, torch2.8, transformers5.10)│
│     /opt/miniconda3/envs/RoboDojo       (py3.10, IsaacSim4.5, sapien3, mplib)│
│                                                                          │
│  ┌────────  Process 1: Policy Server  ────┐  ┌───── Process 2: Env Client ─────┐│
│  │  conda activate holobrain                │  │  conda activate RoboDojo         ││
│  │  CUDA_VISIBLE_DEVICES=0                  │  │  CUDA_VISIBLE_DEVICES=1          ││
│  │                                          │  │                                  ││
│  │  python XPolicyLab/setup_policy_server.py│  │  python src/eval_client/main.py  ││
│  │    --config_path deploy.yml              │  │    --port $PORT                  ││
│  │    --overrides ...                       │  │    --policy_server_url \        ││
│  │                                          │  │        ws://localhost:$PORT     ││
│  │  加载 HoloBrainProcessor + Model         │  │  启动 IsaacSim (headless)        ││
│  │  起 WebSocket server on ws://:$PORT      │◄─┼─ 通信：msgpack + msgpack_numpy   ││
│  │  handlers: HELLO / RESET / INFER / CLOSE │  │                                  ││
│  │                                          │  │  loop: reset → obs → infer → act ││
│  │  收到 INFER：                             │  │        (25 ep/task, 54 task)     ││
│  │    obs → HoloBrain forward               │  │                                  ││
│  │    → return actions (chunk of 4)         │  │  产物：_result.json + episode.mp4││
│  └──────────────────────────────────────────┘  └──────────────────────────────────┘│
│                                                                                    │
│  两进程共享的：                                                                     │
│  - PYTHONPATH=${WORKING_PATH}/robo_orchard_lab:$PYTHONPATH  (robo_orchard_lab)     │
│  - /horizon-bucket/ (fuse mount)                                                   │
│  - ${WORKING_PATH}/{eval_result, smoke_results, ckpt, urdf, Assets}  (bucket symlink)│
└────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 为什么必须两个 env

| 依赖 | holobrain env | RoboDojo env | 冲突原因 |
|---|---|---|---|
| Python | 3.11 | **3.10** | IsaacSim 4.5 硬绑 py3.10（Kit 用 .kit 文件里的 Python interpreter path 固化） |
| numpy | 2.x 兼容 | **< 2.0** | mplib 0.2.1 用 pybind11 编译时 pin numpy 1.x |
| transformers | **5.10.2** | 不需要 | Qwen2.5-VL 加载需要新版；老 sim env 装了会跟其他包冲突 |
| torch | 2.8/cu128 | 2.x (随 IsaacSim) | policy 侧要 flash-attn 兼容；sim 侧不 care |
| sapien / mplib | 不需要 | **必须** | 只 sim 用 |
| IsaacSim / IsaacLab | 不需要 | **必须** | ~10 GB 大依赖 |

结论：分家是**依赖硬约束**，不是设计洁癖。

---

## 3. `submit_cfg` 里做了什么（重点）

以下摘自 `robodojo_pipeline/03_eval.md` §1.2，逐块解释「双 env 相关」的部分：

```bash
# 阶段 A: 基础环境
source /opt/miniconda3/etc/profile.d/conda.sh       # 让 `conda` 可用
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
export TMPDIR=/tmp/isaaclab_kun

# 阶段 B: sim env 找 policy env 的 libpython.so（有些 IsaacSim extension 会 dlopen）
export LD_LIBRARY_PATH=/home/users/kun01.wu-labs/miniconda3/envs/RoboDojo/lib:$LD_LIBRARY_PATH

# 阶段 C: PYTHONPATH 让两个 env 都能 import robo_orchard_lab
export PYTHONPATH=${WORKING_PATH}:${WORKING_PATH}/robo_orchard_lab:$PYTHONPATH

# 阶段 D: symlink bucket → 让 policy 加载 ckpt / URDF
ln -sfn /horizon-bucket/robot_lab/users/kun01.wu/datasets/RoboDojo/Assets Assets
ln -sfn /horizon-bucket/robot_lab2/datasets/all_data/urdf/urdf_v20260711 urdf
ln -sfn /horizon-bucket/robot_lab/users/xuewu.lin/ckpt ckpt
ln -sfn /horizon-bucket/robot_lab/users/kun01.wu/aidi_output/holobrain_robodojo_posttrain_v9/checkpoint_20000 \
        XPolicyLab/policy/HoloBrain/checkpoints/checkpoint_20000

# 阶段 E: 只做一次的 env 修补（改 python 脚本 / sed 打 patch）
conda run -n RoboDojo python utils/update_embodiment_config_path.py
sed -i 's|urdf" = {version = "2.4.31", exact = true}|urdf" = {}|' ...

# 阶段 F: **默认激活 sim env**
conda activate RoboDojo

# 阶段 G: 起后台 rsync 保险丝 + trap EXIT flush
(while true; do
   rsync -a smoke_results/ $OUT_DIR/smoke_results/ 2>/dev/null || true
   rsync -a eval_result/   $OUT_DIR/eval_result/   2>/dev/null || true
   sleep 60
 done) &
BG_RSYNC_PID=$!
trap 'kill $BG_RSYNC_PID 2>/dev/null || true;
      rsync -a smoke_results/ $OUT_DIR/smoke_results/ 2>/dev/null || true;
      rsync -a eval_result/   $OUT_DIR/eval_result/   2>/dev/null || true' EXIT

# 阶段 H: 主命令（在 RoboDojo env 里跑）
bash scripts/robodojo.sh benchmark \
    --policy-dir XPolicyLab/policy/HoloBrain \
    --ckpt checkpoint_20000 \
    --env-cfg arx_x5_holobrain \
    --seed 0 \
    --policy-gpu 0 --env-gpu 1 \        # ← 关键：GPU 隔离
    --policy-env holobrain --eval-env RoboDojo \   # ← 关键：告诉脚本两 env 的名字
    --eval-num 25
```

**核心机制**：`cmd` 里默认激活 `RoboDojo`（因为主 driver 是 `robodojo.sh` + `main.py` 都在 sim 侧）；**policy server 端在子进程里再切**（下面 §4 讲）。

---

## 4. Policy server 端如何切到 holobrain env

关键文件：`~/git_repo/RoboDojo/XPolicyLab/policy/HoloBrain/setup_eval_policy_server.sh`

```bash
#!/bin/bash
# args: bench_name, task, ckpt, env_cfg, action_type, seed, policy_gpu, policy_env, PORT

# L41-42: 切 env
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate ${policy_env}                        # = "holobrain"

# L45-47: 补齐 PYTHONPATH（因为 conda activate 会 reset PYTHONPATH）
export PYTHONPATH="${BENCH_ROOT}/robo_orchard_lab:${PYTHONPATH}"
export HF_HUB_OFFLINE=1

# L49-64: exec (取代当前进程)
exec env CUDA_VISIBLE_DEVICES=${policy_gpu} python XPolicyLab/setup_policy_server.py \
     --config_path deploy.yml \
     --overrides port=${PORT} host=localhost \
                 bench_name=... task_name=... ckpt_name=checkpoint_20000 \
                 env_cfg_type=arx_x5_holobrain seed=0 policy_name=HoloBrain \
                 action_type=joint action_dim=...
```

这个 `.sh` 由 `run_policy_eval.sh` **后台启动**：

```bash
# run_policy_eval.sh:94-108（简化）
bash XPolicyLab/policy/HoloBrain/setup_eval_policy_server.sh \
    $bench_name $task $ckpt $env_cfg $action_type $seed $policy_gpu $policy_env $PORT &
SERVER_PID=$!

# run_policy_eval.sh:110-115 wait for server
bash XPolicyLab/utils/wait_for_policy_server.sh $PORT 600

# run_policy_eval.sh:117-130 起 env client
bash XPolicyLab/policy/HoloBrain/setup_eval_env_client.sh ...
```

**关键点**：
- 用 `bash sub-script &` 后台起 server，父 shell 继续
- 子 shell 里 `conda activate holobrain` **不影响父 shell**（父 shell 仍在 `RoboDojo` env）
- `exec env CUDA_VISIBLE_DEVICES=0 python ...` 让 server 进程只看得到 GPU 0
- `wait_for_policy_server.sh` 循环 `nc -z localhost $PORT`，就绪才起 client
- Client 用 `--env-gpu 1` → env `CUDA_VISIBLE_DEVICES=1`

---

## 5. 通信协议

**Transport**: WebSocket (`ws://localhost:$PORT`)
**Codec**: msgpack + msgpack_numpy（numpy array 序列化为 `{ndim, dtype, shape, raw bytes}`）

### 5.1 Frame schema (`XPolicyLab/client_server/ws/protocol/schemas.py:31`)

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
    payload: dict               # obs / action / status
```

### 5.2 消息流

```
Client (env)                                     Server (policy)
    │                                                  │
    │  HELLO { version, capabilities }                │
    ├──────────────────────────────────────────────►  │
    │                                          HELLO_ACK
    │  ◄──────────────────────────────────────────────┤
    │                                                  │
    │  RESET (per episode start)                       │
    ├──────────────────────────────────────────────►  │
    │                                     RESET_RESULT (model.reset())
    │  ◄──────────────────────────────────────────────┤
    │                                                  │
    │  INFER { payload.observation }                   │
    ├──────────────────────────────────────────────►  │
    │                                    (update_obs + get_action)
    │                                    INFER_RESULT { payload.actions[chunk_of_4] }
    │  ◄──────────────────────────────────────────────┤
    │                                                  │
    │  ... 每 step 1 次 INFER, chunk_size=4 复用 4 步 ...│
    │                                                  │
    │  CLOSE (episode end / task done)                 │
    ├──────────────────────────────────────────────►  │
```

**Obs / Action 结构**详见 `robodojo_pipeline/03_eval.md` §5。

---

## 6. GPU 隔离机制

Pod 内 8 张 GPU：

| GPU | 谁用 | 怎么控制 |
|---|---|---|
| GPU 0 | Policy server | `setup_eval_policy_server.sh` 里 `env CUDA_VISIBLE_DEVICES=$policy_gpu`（`--policy-gpu 0`） |
| GPU 1 | Env client (IsaacSim) | `main.py --device_id 1` + `AppLauncher --device_id 1` |
| GPU 2-7 | 闲置 | 评测不需要 |

**为什么不合并到一张卡**：
- HoloBrain 3B VLM + diffusion head ≈ 8-10 GB
- IsaacSim + scene assets ≈ 6-12 GB
- 单卡容易 OOM，双卡稳
- 5090 32GB 单卡技术上够，但双卡跑更保险

**为什么不减 `gpu_per_worker=2`**：
- 集群 5090 队列的最小 unit 常是 8 卡机（`gpu_per_worker=8`），减到 2 反而排队更久
- 剩下 6 卡闲置无所谓，钱按 pod 计

---

## 7. 生命周期时序（一次完整评测 pod）

```
t=0    AIDI 起 pod，mount bucket，解 tar
       bash $WORKING_PATH/run.sh
        └── bash run_local.sh   (num_workers=1，无 get_rank)
             └── [ 阶段 A-G 的 cmd 依次执行 ]
                 conda activate RoboDojo    ← 此后主进程在 RoboDojo env
                 bg rsync loop 起来
                 bash scripts/robodojo.sh benchmark ...

t=+5min ─ smoke_all_tasks.sh 拿到 54 task 列表
                              │
                              ▼
       for task in TASKS:
         bash scripts/robodojo.sh eval --task $task
           └── run_policy_eval.sh
                ├── PORT=$(get_free_port)
                ├── trap cleanup EXIT INT TERM
                ├── bash setup_eval_policy_server.sh $task ... &      ← 后台
                │     └── conda activate holobrain
                │         exec env CUDA_VISIBLE_DEVICES=0 python XPolicyLab/setup_policy_server.py
                │             └── 加载 HoloBrainProcessor + Model
                │                 起 ws server on :$PORT
                │                 asyncio.run(serve_forever)
                ├── wait_for_policy_server.sh $PORT 600
                └── bash setup_eval_env_client.sh $task ...
                      └── env CUDA_VISIBLE_DEVICES=1 python src/eval_client/main.py \
                            --port $PORT --policy_server_url ws://localhost:$PORT
                          └── AppLauncher → IsaacSim boot
                              EvalEnv(env_cfg) → 建 4 大 manager (robot/scene/camera/reward)
                              while env.env_seeds is not None:
                                 env.reset(seed=...)           ── send RESET
                                 env.run_eval()                ── loop send INFER, receive actions
                              print "Success nums: X, Fail nums: Y"
                              ── write _result.json (直落 bucket via symlink)
                              ── write episode_*.mp4
                      env client 正常退出 rc=0
                └── kill $SERVER_PID; wait
       record_result $task $rc ...  → smoke_results/<run_id>.{json,md} 追加
       [ 下一 task ]

t=+30h  ─ 或跑完 54 task 或 wall_time=2880 SIGTERM 到期
       trap EXIT 触发：
         kill $BG_RSYNC_PID
         最后一次 rsync flush eval_result/ smoke_results/ → bucket
       run_policy_eval.sh trap cleanup 触发：
         _kill_process_tree $SERVER_PID  (TERM 然后 KILL)
       pod 收尾，AIDI /job_data → output/
```

---

## 8. 「一个集群任务里跑两个 conda env」的通用配方

不限于 HoloBrain × RoboDojo，只要评测/推理任务需要多 env，都可套用：

### 步骤 1：镜像预置多个 env
- 起 dev container 装 base env A
- 再 `conda create -n B python=X` 装 env B
- `docker commit + push` 出镜像

### 步骤 2：submit_cfg.json 里的 cmd 模式
```json
{
  "python_launcher": "python3",
  "python_executable": null,          // 或整段丢到 cmd 里
  "cmd": [
    "set -euo pipefail",
    "source /opt/miniconda3/etc/profile.d/conda.sh",
    "export ...",
    "ln -sfn /horizon-bucket/... assets",
    "conda activate <PRIMARY_ENV>",   // 主 driver 用的 env
    "bash <driver_script>.sh"          // driver 内部再起后台子进程切次要 env
  ]
}
```

### 步骤 3：driver script 起后台服务
```bash
# driver.sh
PORT=$(get_free_port.sh)

# 后台启动 secondary env 的 server
(
  source /opt/miniconda3/etc/profile.d/conda.sh
  conda activate <SECONDARY_ENV>
  exec env CUDA_VISIBLE_DEVICES=$SECONDARY_GPU python secondary_server.py --port $PORT
) &
SERVER_PID=$!

trap "kill $SERVER_PID 2>/dev/null; wait" EXIT INT TERM

# 等 server ready
until nc -z localhost $PORT; do sleep 1; done

# 前台跑 primary env 的 client
env CUDA_VISIBLE_DEVICES=$PRIMARY_GPU python primary_client.py --url ws://localhost:$PORT

# 显式停 server
kill $SERVER_PID; wait
```

### 步骤 4：communicate
- **WebSocket + msgpack**（本项目走这条）
- 或 gRPC / HTTP / Unix socket / shared memory
- HoloBrain 用 msgpack_numpy 直接串 numpy array，避免 base64 编码开销

### 关键 pitfalls
- **子 shell 的 `conda activate` 不影响父 shell** —— 这正是我们要的
- **`conda activate` 会重置 PYTHONPATH** —— 切完后重新 export
- **`exec` 让子进程替代 shell 进程** —— 信号能穿透（父 kill server 直接杀到 python）
- **GPU 隔离要用 `CUDA_VISIBLE_DEVICES`**，不要用 `torch.cuda.set_device()`（那是进程内切换，一样争 memory pool）
- **端口冲突**：用 `get_free_port.sh` 动态分，别硬编码
- **超时 wait**：`wait_for_policy_server.sh $PORT 600`，等 10 min 起不来就放弃

---

## 9. 常见问题

### Q: 两个 env 都要 `robo_orchard_lab`，装两份？
- 不用。用 `PYTHONPATH=${WORKING_PATH}/robo_orchard_lab:$PYTHONPATH` 让两个 env 都能 import
- 前提是两个 env 都装了 `robo_orchard_lab` **需要的 base dep**（如 pydantic、numpy）
- 具体见 memory `[[robodojo-holobrain-eval-image-v6]]`

### Q: `conda run -n X python foo.py` vs `conda activate X && python foo.py`
- `conda run` 适合**一次性命令**，不改父 shell 状态
- `conda activate` 改父 shell，之后所有命令都在该 env
- 本项目 `update_embodiment_config_path.py` 用 `conda run`（一次性），主 driver 用 `conda activate`

### Q: 一个 pod 起两个 sim env（都要 IsaacSim）行吗？
- 理论行，但 IsaacSim 4.5 每个实例吃 6-12 GB VRAM + 2-4 CPU cores，8 卡 pod 只能起 4 个左右
- 更简单的：一个 sim × 多 seed 复用同一 IsaacSim 实例（seed_manager 循环）

### Q: 为什么不用 subprocess + Pipe 通信？
- pipe 不能跨进程共享 numpy buffer；每次 pickle+unpickle 慢
- WebSocket + msgpack 是**跨 env / 跨机 / 跨语言**通用方案
- 未来若把 policy server 拆到别的 pod（甚至别的镜像），只改 URL 就行

---

下一篇 [05_faq_and_hidden_gotchas.md](05_faq_and_hidden_gotchas.md) 是集大成的坑清单。
