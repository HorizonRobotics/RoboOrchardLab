#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# eval_robotwin_ckpt11.sh
#
# 用备份出来的 checkpoint_11 (step ≈ 60000) 在 RoboTwin 2.0 上做本机评估。
#
# 前置条件：
#   1. checkpoint_11 已备份到 workspace/checkpoints_backup/checkpoint_11_step60000/
#      并且 EVAL_MODEL_DIR (checkpoint_11_eval) 已经组装好。若尚未组装，本脚
#      本会自动补上（幂等）。
#   2. 已经激活好 RoboTwin 自己的 conda 环境（依赖 torch==2.4.1 / sapien 3.0.0b1
#      / mplib 0.2.1，与 holobrain_internal 冲突，请勿在 holobrain_internal
#      里跑）。
#   3. GPU 2/3 空闲（GPU 0 上仍在训练 HoloBrain，不能碰）。
#
# 用法：
#   bash projects/holobrain_internal/common/scripts/eval_robotwin_ckpt11.sh
#
# 可通过环境变量覆盖：
#   ROBOTWIN_DIR       默认: /horizon-bucket/robot_lab/users/xuewu.lin/self-collected-data/robotwin
#   TASK_NAMES         默认: place_empty_cup,stack_blocks_three
#   TASK_CONFIG        默认: demo_clean
#   TEST_NUM           默认: 100
#   CUDA_DEVICES       默认: 2,3
#   RW_ROOT            默认: $HOME/robotwin_eval_run  (仅当 ROBOTWIN_DIR 只读时启用)
# ---------------------------------------------------------------------------

set -euo pipefail

# ------- 路径常量 -------------------------------------------------------------
REPO_ROOT="/home/users/kun01.wu-labs/git_repo/robo_orchard_lab"
COMMON_DIR="$REPO_ROOT/projects/holobrain_internal/common"
WS="$COMMON_DIR/workspace"
CKPT_SRC="$WS/checkpoints/checkpoint_11"
CKPT_BAK_DIR="$WS/checkpoints_backup"
CKPT_BAK="$CKPT_BAK_DIR/checkpoint_11_step60000"
EVAL_MODEL_DIR="$CKPT_BAK_DIR/checkpoint_11_eval"

# ------- 可覆盖参数 -----------------------------------------------------------
ROBOTWIN_DIR="${ROBOTWIN_DIR:-/horizon-bucket/robot_lab/users/xuewu.lin/self-collected-data/robotwin}"
TASK_NAMES="${TASK_NAMES:-place_empty_cup,stack_blocks_three}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
TEST_NUM="${TEST_NUM:-100}"
CUDA_DEVICES="${CUDA_DEVICES:-2,3}"
RW_ROOT="${RW_ROOT:-$HOME/robotwin_eval_run}"

VLM_CKPT_DIR="/horizon-bucket/robot_lab/users/xuewu.lin/ckpt"
URDF_DIR="/horizon-bucket/robot_lab2/datasets/all_data/urdf/urdf_v20260711"

TS="$(date +%Y%m%d_%H%M%S)"
DRIVER_LOG="$CKPT_BAK_DIR/eval_ckpt11_${TS}.log"

echo "======================================================================"
echo "HoloBrain checkpoint_11 → RoboTwin 2.0 eval"
echo "  ROBOTWIN_DIR = $ROBOTWIN_DIR"
echo "  TASK_NAMES   = $TASK_NAMES"
echo "  TASK_CONFIG  = $TASK_CONFIG"
echo "  TEST_NUM     = $TEST_NUM"
echo "  CUDA         = $CUDA_DEVICES"
echo "  driver log   = $DRIVER_LOG"
echo "======================================================================"

# ------- 0) 前置检查 ----------------------------------------------------------
echo "[0/5] pre-flight checks ..."

[[ -d "$ROBOTWIN_DIR/envs" ]] || { echo "ERR: $ROBOTWIN_DIR/envs missing"; exit 1; }
[[ -f "$ROBOTWIN_DIR/script/eval_policy.py" ]] || { echo "ERR: script/eval_policy.py missing"; exit 1; }
[[ -f "$ROBOTWIN_DIR/task_config/${TASK_CONFIG}.yml" ]] || { echo "ERR: task_config/${TASK_CONFIG}.yml missing"; exit 1; }

# 强烈警告：不要在 holobrain_internal env 里跑
if [[ "${CONDA_DEFAULT_ENV:-}" == "holobrain_internal" ]]; then
    echo "ERR: current conda env is 'holobrain_internal'."
    echo "     RoboTwin 需要独立环境 (torch==2.4.1 + sapien 3.0.0b1 + mplib 0.2.1)。"
    echo "     请先: conda deactivate && conda activate <你的 robotwin env>"
    exit 1
fi

# 冒烟测试：确认当前 env 同时能 import robotwin sim + holobrain model 侧依赖
echo "  smoke-testing imports (robotwin + holobrain in same env) ..."
PYTHONPATH="$ROBOTWIN_DIR:$REPO_ROOT:${PYTHONPATH:-}" \
python3 -c "
import sys
missing = []
for mod in ['sapien', 'mplib', 'gymnasium', 'torch', 'transformers',
            'accelerate', 'safetensors', 'diffusers',
            'robo_orchard_lab.models.holobrain.processor',
            'robo_orchard_lab.models.holobrain.structure',
            'robo_orchard_lab.models.mixin']:
    try:
        __import__(mod)
    except Exception as e:
        missing.append(f'{mod}: {type(e).__name__}: {e}')
if missing:
    print('MISSING/BROKEN imports:', *missing, sep='\n  ')
    sys.exit(1)
print('  all required modules importable')
" || {
    echo ""
    echo "ERR: 当前 conda env 缺少必要包。请在同一个 env 里补装 HoloBrain 侧依赖："
    echo "     pip install 'transformers>=4.49,<4.58' accelerate safetensors diffusers \\"
    echo "                 pydantic filelock requests einops pytorch_kinematics"
    echo "     另外 pytorch3d 与 flash-attn 需按 torch/cuda 版本单独装。"
    exit 1
}

# GPU 空闲检查（只看目标 GPU；默认阈值 1024 MiB，可通过 GPU_FREE_THRESHOLD 放宽）
: "${GPU_FREE_THRESHOLD:=1024}"
for gid in ${CUDA_DEVICES//,/ }; do
    used_mib=$(nvidia-smi -i "$gid" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo "-1")
    if [[ "$used_mib" == "-1" ]]; then
        echo "ERR: 无法查询 GPU $gid 状态，nvidia-smi 是否可用？"
        exit 1
    fi
    if (( used_mib > GPU_FREE_THRESHOLD )); then
        echo "ERR: GPU $gid 显存占用 ${used_mib} MiB > ${GPU_FREE_THRESHOLD} MiB，可能已被其他进程占用。"
        echo "     若确认要在共享 GPU 上跑，重跑时设 GPU_FREE_THRESHOLD=<更大的 MiB 数>。"
        exit 1
    fi
    echo "  GPU $gid usable (used=${used_mib} MiB, threshold=${GPU_FREE_THRESHOLD} MiB)"
done

# ------- 1) 备份 checkpoint_11 ------------------------------------------------
echo "[1/5] backing up checkpoint_11 (skip if already done) ..."
if [[ ! -f "$CKPT_BAK/model.safetensors" ]]; then
    [[ -d "$CKPT_SRC" ]] || { echo "ERR: $CKPT_SRC missing"; exit 1; }
    mkdir -p "$CKPT_BAK_DIR"
    cp -a --reflink=auto "$CKPT_SRC" "${CKPT_BAK}.tmp"
    mv "${CKPT_BAK}.tmp" "$CKPT_BAK"
    echo "  backup created at $CKPT_BAK"
else
    echo "  backup already exists at $CKPT_BAK (mtime $(stat -c '%y' "$CKPT_BAK/model.safetensors"))"
fi

# ------- 2) 组装 EVAL_MODEL_DIR -----------------------------------------------
echo "[2/5] assembling EVAL_MODEL_DIR ..."
mkdir -p "$EVAL_MODEL_DIR"
for src in \
    "$CKPT_BAK/model.safetensors" \
    "$CKPT_BAK/model.config.json" \
    "$WS/robotwin2_0_processor.json" \
    "$WS/robotwin2_0_inference.config.json"
do
    [[ -f "$src" ]] || { echo "ERR: required file missing: $src"; exit 1; }
    cp -a --reflink=auto "$src" "$EVAL_MODEL_DIR/"
done
if [[ ! -d "$EVAL_MODEL_DIR/urdf" ]]; then
    cp -a --reflink=auto "$WS/urdf" "$EVAL_MODEL_DIR/"
fi
echo "  EVAL_MODEL_DIR ready: $EVAL_MODEL_DIR"
ls -la "$EVAL_MODEL_DIR"

# ------- 3) 拷贝 holobrain policy / eval driver 到 ROBOTWIN_DIR ----------------
echo "[3/5] staging holobrain policy into RoboTwin repo ..."

# 若 ROBOTWIN_DIR 只读，rsync 到本地可写副本
if ! touch "$ROBOTWIN_DIR/.write_probe" 2>/dev/null; then
    echo "  $ROBOTWIN_DIR is read-only, rsyncing to $RW_ROOT ..."
    mkdir -p "$RW_ROOT"
    # -a 保留属性；--delete 保持与源同步；--exclude 剪掉体积大且无用的目录
    rsync -a --delete \
        --exclude='data/' \
        --exclude='eval_result/' \
        --exclude='cache—data/' \
        --exclude='log/' \
        --exclude='sem_eval_model/' \
        "$ROBOTWIN_DIR/" "$RW_ROOT/"
    ROBOTWIN_DIR="$RW_ROOT"
    echo "  now using ROBOTWIN_DIR=$ROBOTWIN_DIR"
else
    rm -f "$ROBOTWIN_DIR/.write_probe"
    echo "  $ROBOTWIN_DIR is writable, use in place"
fi

cp -r "$COMMON_DIR/holobrain_robotwin_policy" "$ROBOTWIN_DIR/"
cp -a "$COMMON_DIR/robotwin_eval.py"          "$ROBOTWIN_DIR/"
cp -a "$COMMON_DIR/holobrain_utils.py"        "$ROBOTWIN_DIR/"
echo "  staged: holobrain_robotwin_policy/, robotwin_eval.py, holobrain_utils.py"

# ------- 4) 启动评估 ----------------------------------------------------------
echo "[4/5] launching robotwin_eval.py on CUDA=${CUDA_DEVICES} ..."
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
# ROBOTWIN_DIR: 让 script/eval_policy.py 能 import envs/policy/holobrain_robotwin_policy
# REPO_ROOT   : 让 deploy_policy.py 能 import robo_orchard_lab.*
export PYTHONPATH="$ROBOTWIN_DIR:$REPO_ROOT:${PYTHONPATH:-}"

pushd "$ROBOTWIN_DIR" > /dev/null

# 头部信息也打进 driver log
{
    echo "=== $(date -Is) eval driver start ==="
    echo "ROBOTWIN_DIR    = $ROBOTWIN_DIR"
    echo "EVAL_MODEL_DIR  = $EVAL_MODEL_DIR"
    echo "TASK_NAMES      = $TASK_NAMES"
    echo "TASK_CONFIG     = $TASK_CONFIG"
    echo "TEST_NUM        = $TEST_NUM"
    echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
    echo "PYTHONPATH      = $PYTHONPATH"
    echo "----"
} | tee -a "$DRIVER_LOG"

python3 robotwin_eval.py \
    --task_names "$TASK_NAMES" \
    --task_config "$TASK_CONFIG" \
    --model_config "$EVAL_MODEL_DIR" \
    --model_processor robotwin2_0_processor \
    --model_prefix model \
    --vlm_ckpt_dir "$VLM_CKPT_DIR" \
    --urdf_dir "$URDF_DIR" \
    --test_num "$TEST_NUM" \
    2>&1 | tee -a "$DRIVER_LOG"

RC=${PIPESTATUS[0]}
popd > /dev/null

# ------- 5) 汇总提示 ----------------------------------------------------------
echo "[5/5] done (robotwin_eval.py returncode=$RC)."
echo ""
echo "  Per-task logs (子进程 stdout):"
for t in ${TASK_NAMES//,/ }; do
    echo "    $ROBOTWIN_DIR/eval_result/$t/$TASK_CONFIG/log.txt"
done
echo ""
echo "  Per-episode 视频与 _result.txt:"
for t in ${TASK_NAMES//,/ }; do
    echo "    $ROBOTWIN_DIR/eval_result/$t/holobrain_robotwin_policy/$TASK_CONFIG/<null>/<timestamp>/"
done
echo ""
echo "  Driver log (含总 JSON 汇总):"
echo "    $DRIVER_LOG"

exit "$RC"
