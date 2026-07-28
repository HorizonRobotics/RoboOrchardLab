# Claude Session Handover — HoloBrain 项目 (v2)

> **日期**：2026-07-22（更新）
> **本 session 结束时状态**：任务 #4 "Run RoboTwin eval" **被 `curobo` 阻塞**，用户决定暂停并做完整交接。
> **交接给**：下一个 Claude session。
>
> 本文档目的：让下一个 Claude 一次读完就能无缝接手完成 checkpoint_11 的 RoboTwin 2.0 评估。

---

## 0. TL;DR — 下次接手直接看这里

**已完成**：
1. `conda env robotwin_holobrain_eval` **已装好** (py3.11)，155 个包，torch 2.8.0+cu128、transformers 5.10.2、sapien 3.0.0b1、mplib 0.2.1、pytorch3d 0.7.9、flash-attn 2.8.3 全对齐训练 env。
2. **模型加载在 GPU 2 上验证过**：`HoloBrain_Qwen2_5_VL` 824 权重成功 `to('cuda')`，占 2.7 GB 显存。
3. **RoboTwin 仓库已本地化到 `~/git_repo/robotwin/`**（rsync 自 bucket，53 MB，assets/data 软链保留）。
4. **RoboTwin `_install.sh` 的 2 处 sed patch 已打**（sapien URDF utf-8、mplib planner 去 collide）。
5. **`holobrain_robotwin_policy` + `robotwin_eval.py` + `holobrain_utils.py` 已 staged 到 `~/git_repo/robotwin/`**。
6. **`EVAL_MODEL_DIR/ckpt`** 软链已建（`ln -sfn /horizon-bucket/.../ckpt EVAL_MODEL_DIR/ckpt`）。
7. **训练进程 PID 1540974 仍 alive (26h+)**，GPU 0 上 ~24 GB 显存。

**阻塞点** —— 需要 curobo：
- `envs/robot/robot.py:15` 硬 `from .planner import CuroboPlanner`（planner.py 里定义 `class CuroboPlanner` 在 try 块内，import curobo 失败则未定义）。
- **curobo 装不上**：GitHub `NVlabs/curobo` 从本机 timeout；hobot mirror 里 `curobo` (0.1/0.2) / `Nvidia-curobo` (0.1) 都是 <1 KB **占位包**（我误装了它们，需要 uninstall）。
- 用户尚未提供内部 curobo 源。

**下一步（下 session 要问用户）**：
1. **询问用户**：有没有内部 gitlab / bucket 里的 curobo 源码路径？如 `/horizon-bucket/.../curobo/` 或 `~/git_repo/curobo/`？或者 hobot art-internal 里的 curobo wheel URL？
2. 或者：**用户允许 fallback**——修改 `~/git_repo/robotwin/envs/robot/robot.py` 让 CuroboPlanner 变可选，缺失时 `left_planner = MplibPlanner(...)`（planner.py:14 已有该 class）。**只改本地副本，不动 bucket 原代码**。

**跑评估的最终命令**（curobo 问题解决后）：
```bash
export ROBOTWIN_DIR=/home/users/kun01.wu-labs/git_repo/robotwin
export CUDA_DEVICES=2
export GPU_FREE_THRESHOLD=10240   # GPU 2 上有 mengchen.ma 的 cosmos-framework 占 8.9 GB
export SAPIEN_HEADLESS=1
export CUDA_HOME=/usr/local/cuda-12.8

source /home/users/kun01.wu-labs/miniconda3/etc/profile.d/conda.sh
conda activate robotwin_holobrain_eval

cd /home/users/kun01.wu-labs/git_repo/robo_orchard_lab
# 先短跑冒烟 (1 task × 3 trial, ~5-10 min)
TASK_NAMES=place_empty_cup TEST_NUM=3 \
    bash projects/holobrain_internal/common/scripts/eval_robotwin_ckpt11.sh
# 然后正式跑 (2 task × 100 trial, ~60-120 min 单卡)
bash projects/holobrain_internal/common/scripts/eval_robotwin_ckpt11.sh
```

---

## 1. 用户身份与环境（对齐 v1 交接）

- 仓库：`/home/users/kun01.wu-labs/git_repo/robo_orchard_lab`（分支 `feature/memory_dev1`）
- 主项目：`projects/holobrain_internal/`
- 平台：Linux + Miniconda（`/home/users/kun01.wu-labs/miniconda3`）
- 硬件：8 × RTX 5090 (sm_120)，CUDA driver 570.211.01, `/usr/local/cuda-12.8` 就位
- 交流：中文，file:line 引用
- 硬约束：**绝不改动 `/horizon-bucket/robot_lab/users/xuewu.lin/self-collected-data/robotwin/`**（xuewu.lin 的仓库，用户明确要求）

## 2. 正在跑的训练进程 — 不能碰

- PID **1540974**，`python3 train.py --config configs/config_holobrain_common.py`
- cwd: `projects/holobrain_internal/common/`
- **GPU 0 独占** ~24 GB，271% CPU
- 产出：`workspace/checkpoints/checkpoint_{9,10,11}/`（accelerate rolling `total_limit=3`）
- **绝对不能**：写 `workspace/checkpoints/`、动 GPU 0、kill 该进程。

## 3. 本 session 完成的核心工作流

### 3.1 conda env `robotwin_holobrain_eval` 装依赖（155 包）

**关键决策纠正**（v1 交接文档里的错误假设）：
- ~~py3.10~~ → **py3.11**（对齐训练 env，直接复用其编译好的 pytorch3d/flash-attn/nvidia-cu12 目录，省 30-60 min nvcc）
- ~~torch 2.4.1~~ → **torch 2.8.0 (PyPI 默认 wheel 就是 cu128 build)**（RTX 5090=sm_120 必须 CUDA 12.8+）
- ~~download.pytorch.org 直连~~ → **hobot mirror `http://pypi.hobot.cc/simple` + 清华兜底**
- ~~安装 Xvfb + ffmpeg~~ → **conda-forge 里没有 `xorg-server-xvfb` 包名，跳过**（sapien 用 `SAPIEN_HEADLESS=1` env var + 系统 `libGL.so.1` 应可跑）

**装依赖的巧办法**（技术亮点，下 session 若要重装可复用）：
1. **训练 env 里已装的大件（nvidia-cu12 系 5.9 GB / pytorch3d / flash-attn / robo_orchard_core）通过 `cp -a --reflink=auto` 复制到新 env**（CoW，0.2 秒）：
   ```bash
   TRAIN_SP=/home/users/kun01.wu-labs/miniconda3/envs/holobrain_internal/lib/python3.11/site-packages
   NEW_SP=/home/users/kun01.wu-labs/miniconda3/envs/robotwin_holobrain_eval/lib/python3.11/site-packages
   cp -a --reflink=auto $TRAIN_SP/nvidia $NEW_SP/
   cp -a --reflink=auto $TRAIN_SP/pytorch3d $NEW_SP/
   cp -a --reflink=auto $TRAIN_SP/pytorch3d-0.7.9.dist-info $NEW_SP/
   cp -a --reflink=auto $TRAIN_SP/flash_attn $NEW_SP/
   cp -a --reflink=auto $TRAIN_SP/flash_attn-2.8.3.dist-info $NEW_SP/
   cp -a --reflink=auto $TRAIN_SP/flash_attn_2_cuda*.so $NEW_SP/
   cp -a --reflink=auto $TRAIN_SP/robo_orchard_core $NEW_SP/
   cp -a --reflink=auto $TRAIN_SP/robo_orchard_core-*.dist-info $NEW_SP/
   for nv in nvidia_cuda_nvrtc_cu12-12.8.93 nvidia_cuda_runtime_cu12-12.8.90 \
             nvidia_cuda_cupti_cu12-12.8.90 nvidia_cudnn_cu12-9.10.2.21 \
             nvidia_cublas_cu12-12.8.4.1 nvidia_cufft_cu12-11.3.3.83 \
             nvidia_curand_cu12-10.3.9.90 nvidia_cusolver_cu12-11.7.3.90 \
             nvidia_cusparse_cu12-12.5.8.93 nvidia_cusparselt_cu12-0.7.1 \
             nvidia_nccl_cu12-2.27.3 nvidia_nvtx_cu12-12.8.90 \
             nvidia_nvjitlink_cu12-12.8.93 nvidia_cufile_cu12-1.13.1.3 \
             nvidia_cuda_cccl_cu12-12.8.90 triton-3.4.0; do
       cp -a --reflink=auto $TRAIN_SP/${nv}.dist-info $NEW_SP/
   done
   ```
2. **从 pip cache 反打包 wheels 到 stage 目录**（`/home/users/kun01.wu-labs/tmp/local_wheels/`，86 个 wheels）：
   - 脚本在 `/home/users/kun01.wu-labs/tmp/stage_wheels.py`
   - 用于**离线安装** torch/transformers/accelerate/diffusers/safetensors 等。
3. **RoboTwin sim 侧从 hobot mirror 直装**：`sapien==3.0.0b1 mplib==0.2.1 gymnasium==0.29.1 transforms3d==0.4.2 trimesh==4.4.3 pyglet<2 termcolor av opencv-python==4.11.0.86 open3d==0.18.0 toppra pyperclip cloudpickle farama-notifications`。
4. **numpy 必须锁 1.26.4**（sapien/mplib 要 numpy<2；`numpydantic 1.8.1` 是唯一 numpy<2 兼容版）：
   ```bash
   pip install --force-reinstall --no-deps 'numpy==1.26.4' 'numpydantic==1.8.1'
   ```

### 3.2 关键 pip mirror 配置

用户明确要求**优先 hobot 内部镜像**，其次清华：

```
--index-url http://pypi.hobot.cc/simple
--extra-index-url http://pypi.hobot.cc/hobot-local/simple
--trusted-host pypi.hobot.cc
```

（**用户 pip.conf 里默认 `index-url` 写的是 `pypi.hobot.cc/hobot-local/simple`——错误，那里没 torch。得显式覆盖为 `pypi.hobot.cc/simple`。**）

### 3.3 sed patches 已应用

- `sapien/wrapper/urdf_loader.py:667,673`: `open(urdf_file, "r")` → `open(urdf_file, "r", encoding="utf-8")` ✓
- `mplib/planner.py:807`: `if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:` → 去掉 `or collide` ✓

### 3.4 checkpoint_11 evaluation 目录（v1 已建，本 session 补 ckpt symlink）

```
workspace/checkpoints_backup/checkpoint_11_eval/
├── model.safetensors                       # 2.8 GB
├── model.config.json                       # accelerate save_state 输出
├── robotwin2_0_processor.json              # 来自 workspace/ 顶层
├── robotwin2_0_inference.config.json       # 同上
├── urdf/                                   # 拷贝自 workspace/urdf/
└── ckpt -> /horizon-bucket/robot_lab/users/xuewu.lin/ckpt   # 本 session 新建, VLM backbone 需要
```

**`ckpt` symlink 是本 session 新加的**——`model.config.json` 里 `vlm_pretrain='./ckpt/Qwen2.5-VL-3B-Instruct'` 是相对 EVAL_MODEL_DIR 的路径。`HoloBrainPolicy.__init__` 也会做同样的 symlink（`deploy_policy.py:99`），但**冒烟测试要预先建**。

### 3.5 RoboTwin 仓库本地副本

```
~/git_repo/robotwin/          # rsync 自 /horizon-bucket/.../robotwin/, 53 MB
├── assets -> /horizon-bucket/robot_lab2/users/tianwei.lin/data/robotwin2/assets  # 外部软链保留
├── data -> /horizon-bucket/robot_lab2/users/xuewu.lin/robotwin2.0/raw_data       # 外部软链保留
├── envs/, script/, task_config/, description/, policy/, sem_robotwin_policy/, ...
├── holobrain_robotwin_policy/   # 本 session 拷贝进来的
├── holobrain_utils.py           # 同上
└── robotwin_eval.py             # 同上
```

**排除**：`data/`, `eval_result/`, `cache—data/`, `log/`, `sem_eval_model/`（evaluator 会自己重建/或不需要）。

**注意**：`~/git_repo/robotwin/` 是本地可写副本，**所有 sed patch / policy staging / envs 修改**都在这里做，**绝不动 bucket**。

### 3.6 已修改的 eval_robotwin_ckpt11.sh

新增 `GPU_FREE_THRESHOLD` env var（默认 1024 MiB，允许覆盖）。原脚本 GPU 空闲 <1 GB 才继续，改为可覆盖阈值。当前 GPU 2 上有 8.9 GB 别人的 cosmos_framework，需 `GPU_FREE_THRESHOLD=10240` 才能跑。

## 4. **⭐ 阻塞点：curobo 装不上**

### 症状

跑 `bash projects/holobrain_internal/common/scripts/eval_robotwin_ckpt11.sh` 时子进程崩：

```
File "/home/users/kun01.wu-labs/git_repo/robotwin/envs/robot/robot.py", line 15
    from .planner import CuroboPlanner
ImportError: cannot import name 'CuroboPlanner' from 'envs.robot.planner' 
```

日志见 `/home/users/kun01.wu-labs/git_repo/robotwin/eval_result/place_empty_cup/demo_clean/log.txt`。

driver 日志：`workspace/checkpoints_backup/eval_ckpt11_20260722_093622.log`（`EVAL_EXIT=1`）。

### 根因

- `envs/robot/robot.py:15` 无条件 `from .planner import CuroboPlanner`。
- `envs/robot/planner.py:170-407` 里 `class CuroboPlanner:` 在 `try: from curobo... except:` 块内。若 `import curobo.*` 失败，class 不定义，`from planner import CuroboPlanner` 就 raise ImportError。
- `envs/robot/robot.py:135, 268, 275, ...` 大量使用 `CuroboPlanner(...)` 和 `isinstance(..., CuroboPlanner)`——**curobo 是硬依赖**。

### 已尝试的解法

1. **hobot mirror `curobo`（0.1/0.2）**：只 900 bytes 占位包，实际无 module ❌
2. **hobot mirror `Nvidia-curobo`（0.1）**：1 KB 占位包 ❌
3. **PyPI `curobo` / `Nvidia-curobo`**：也是 squatter ❌
4. **GitHub `NVlabs/curobo`**：本机 timeout ❌
5. **国内 gitee / gitcode mirror**：timeout ❌

### 已装的假 curobo（下 session 若要走 fallback 方案先删）

```bash
pip uninstall -y curobo Nvidia-curobo
```

### 下 session 决策路径

**A. 用户提供 curobo 源** → 装:
   - 若本地路径：`pip install -e /path/to/curobo --no-build-isolation`（nvcc 编译 30-60 min，需要 `CUDA_HOME=/usr/local/cuda-12.8`）
   - 若 hobot art-internal wheel：`pip install <URL>`

**B. Fallback：改本地 `~/git_repo/robotwin/envs/robot/robot.py`**（不动 bucket）：
   - 把 `from .planner import CuroboPlanner` 包 try/except，缺失时 `CuroboPlanner = None`。
   - 改 `robot.py:135, 268, 275, ...` 里所有 `isinstance(planner, CuroboPlanner)` 和 `CuroboPlanner(...)` 逻辑，缺失时用 `MplibPlanner`。
   - **风险**：MplibPlanner 是 mplib.sapien_utils.SapienPlanner 的封装，与 CuroboPlanner 的规划质量/接口不完全等价。可能导致比训练时低的 success rate（不利于跟其他 checkpoint 公平对比），但**能得到一个可参考的 Success rate**。

**C. 用户提供 nvcc 编译好的 curobo binary**（比如别人的 site-packages）：
   - 类似 pytorch3d/flash_attn 的做法，`cp -a --reflink=auto` 过来。
   - 需要用户告知源 site-packages 路径。

## 5. 关键路径速查

```bash
# 仓库
REPO_ROOT=/home/users/kun01.wu-labs/git_repo/robo_orchard_lab
COMMON_DIR=$REPO_ROOT/projects/holobrain_internal/common
WS=$COMMON_DIR/workspace

# checkpoint
$WS/checkpoints/checkpoint_{9,10,11}/       # 训练滚动窗口, 别碰
$WS/checkpoints_backup/checkpoint_11_step60000/     # 冷备份 (v1 建)
$WS/checkpoints_backup/checkpoint_11_eval/          # 组装用于 evaluator (v1 建, 本 session 加 ckpt symlink)

# RoboTwin 本地副本 (本 session 新建)
ROBOTWIN_DIR=/home/users/kun01.wu-labs/git_repo/robotwin

# 外部资源
VLM_CKPT_DIR=/horizon-bucket/robot_lab/users/xuewu.lin/ckpt      # 里面有 Qwen2.5-VL-3B-Instruct/
URDF_DIR=/horizon-bucket/robot_lab2/datasets/all_data/urdf/urdf_v20260711

# conda envs
holobrain_internal          # 训练用, py3.11, 别动
robotwin_holobrain_eval     # 评估用融合 env, py3.11, 155 个包 (本 session 装完)

# 工具
CUDA_HOME=/usr/local/cuda-12.8   # nvcc 12.8, sm_120 needed
TMPDIR=/home/users/kun01.wu-labs/tmp    # /tmp 只剩 21 GB, /home 剩 3.6 T
```

## 6. GPU 占用情况（评估必须避开）

截至本 session 结束：
```
GPU 0: 训练进程 1540974 独占 (24 GB) — 别碰
GPU 1: zhengyu.zou-labs 训练 (25 GB) — 别碰
GPU 2: mengchen.ma-labs cosmos-framework 8.9 GB — 尚有 23 GB, 可用
GPU 3: zhengyu.zou-labs 训练 (29 GB) — 别碰
GPU 4-7: 有 zhengyu.zou-labs 训练 (10-22 GB 不等) — 别碰
```

**唯一可用**：GPU 2（`CUDA_DEVICES=2`），配合 `GPU_FREE_THRESHOLD=10240` 绕过脚本的 <1 GB 检查。

## 7. 环境变量速查（每次跑评估都要 export）

```bash
export ROBOTWIN_DIR=/home/users/kun01.wu-labs/git_repo/robotwin
export CUDA_DEVICES=2                        # 单卡 (只有 GPU 2 可用)
export GPU_FREE_THRESHOLD=10240              # 允许 GPU 上有 <=10 GB 占用
export SAPIEN_HEADLESS=1                     # 无 X11
export CUDA_HOME=/usr/local/cuda-12.8        # nvcc 12.8 与 torch cu128 匹配
export TMPDIR=/home/users/kun01.wu-labs/tmp  # /tmp 只 21 GB free
export PIP_CACHE_DIR=/home/users/kun01.wu-labs/.cache/pip

source /home/users/kun01.wu-labs/miniconda3/etc/profile.d/conda.sh
conda activate robotwin_holobrain_eval
```

## 8. 冒烟测试脚本（下 session 快速验证 env 完好）

```bash
CUDA_VISIBLE_DEVICES="" /home/users/kun01.wu-labs/miniconda3/envs/robotwin_holobrain_eval/bin/python <<'PY'
mods = ['numpy','sapien','mplib','gymnasium','trimesh','transforms3d','pyglet','av','open3d','cv2','iopath','fvcore','moviepy','h5py','torch','torchvision','transformers','accelerate','safetensors','diffusers','pydantic','einops','flash_attn','pytorch3d','pytorch_kinematics','rtoml','robo_orchard_core','numpydantic','toppra','pyperclip','cloudpickle','farama_notifications','deprecated','filelock','requests','jinja2','fsspec','tqdm','yaml','matplotlib','scipy','PIL','datasets','sqlalchemy','pyarrow','lmdb','backports.zstd']
ok = 0; err = []
for m in mods:
    try:
        v = __import__(m); ok += 1
    except Exception as e:
        err.append((m, e))
print(f'OK: {ok}/{len(mods)}')
for m, e in err:
    print(f'  ERR {m}: {type(e).__name__}: {e}')
PY
```

期望：全部 OK（46/46）。

## 9. 模型加载冒烟（GPU 2）

```bash
export CUDA_VISIBLE_DEVICES=2
export SAPIEN_HEADLESS=1
/home/users/kun01.wu-labs/miniconda3/envs/robotwin_holobrain_eval/bin/python <<'PY'
import os, sys, torch
sys.path.insert(0, '/home/users/kun01.wu-labs/git_repo/robo_orchard_lab/projects/holobrain_internal/common')
from robo_orchard_lab.models.mixin import ModelMixin
from robo_orchard_lab.models.holobrain.processor import HoloBrainProcessor
D = '/home/users/kun01.wu-labs/git_repo/robo_orchard_lab/projects/holobrain_internal/common/workspace/checkpoints_backup/checkpoint_11_eval'
proc = HoloBrainProcessor.load(D, 'robotwin2_0_processor.json')
model = ModelMixin.load_model(D, load_impl='native')
model = model.eval().to('cuda')
print(f'OK model={type(model).__name__} mem={torch.cuda.memory_allocated()//2**20} MiB')
PY
```

期望：`OK model=HoloBrain_Qwen2_5_VL mem=2730 MiB`（当前 verified）。

## 10. TaskList 快照

```
#1  [completed] Install fused deps in robotwin_holobrain_eval
#2  [completed] Smoke test imports  (46/46)
#3  [completed] Load model smoke test on GPU 2  (2.7 GB, HoloBrain_Qwen2_5_VL)
#4  [blocked]   Run RoboTwin eval on GPU 2  (curobo 不可装, 需用户决策)
```

## 11. 相关文件位置速查

| 内容 | 路径 |
|---|---|
| Plan 文件 (本 session 主) | `/home/users/kun01.wu-labs/.claude/plans/breezy-floating-star.md` |
| Plan 文件 (v1) | `/home/users/kun01.wu-labs/.claude/plans/crystalline-conjuring-stearns.md` |
| v1 交接文档 | 此前的 `projects/holobrain_internal/docs/claude_tasks.md`（**本文件替换**并移入 `claude_tasks/` 子目录）|
| eval 脚本 | `projects/holobrain_internal/common/scripts/eval_robotwin_ckpt11.sh` |
| eval 说明 | `projects/holobrain_internal/docs/eval_robotwin_ckpt11.md` |
| stage wheels 脚本 | `/home/users/kun01.wu-labs/tmp/stage_wheels.py` |
| stage wheels 目录 | `/home/users/kun01.wu-labs/tmp/local_wheels/` (86 wheels) |
| install logs | `/home/users/kun01.wu-labs/tmp/install_logs/` |
| driver log (失败) | `workspace/checkpoints_backup/eval_ckpt11_20260722_093622.log` |
| Memory 索引 | `~/.claude/projects/-home-users-kun01-wu-labs-git-repo-robo-orchard-lab/memory/MEMORY.md` |

## 12. 关键 memory 条目（本 session 新增）

- `robotwin-eval-env-uses-py311.md`：融合 env 用 py3.11、PyPI 默认 torch 2.8 已是 cu128 build。
- `robotwin-repo-local-copy.md`：RoboTwin 仓库 rsync 到 `~/git_repo/robotwin/`，绝不改动 bucket 原代码。

---

## 给下一个 Claude 的一句话摘要

> **训练进程 PID 1540974 在 GPU 0 上跑着别碰**；checkpoint_11 已备份 + eval 目录组装完 + ckpt symlink 就位；`robotwin_holobrain_eval` env 装了 155 个包（sapien/mplib/torch2.8cu128/transformers5.10.2/pytorch3d/flash-attn 全就位，46/46 冒烟 OK，模型能 to cuda 用 2.7 GB）；**唯一阻塞**是 `curobo` 装不上，需要**问用户要 curobo 源**或者**允许 fallback 改 `~/git_repo/robotwin/envs/robot/robot.py` 用 MplibPlanner**。之后只要用 GPU 2 跑 `bash projects/holobrain_internal/common/scripts/eval_robotwin_ckpt11.sh`（env vars 见 §7）。

---
**本文档路径**：`projects/holobrain_internal/docs/claude_tasks/2026-07-22_robotwin_eval_env_ready_blocked_curobo.md`

**同目录下的历史交接**：按日期顺序累积，最新的在最后。下次接手时读**日期最新**且**未标记 `_resolved`** 的那一份。
