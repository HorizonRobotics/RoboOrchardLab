#!/usr/bin/env bash
# Run robodojo_eval.py on the dev box instead of in an AIDI pod.
#
# WHY THIS EXISTS AS A SEPARATE FILE
#
# robodojo_eval.py's defaults are pod paths, and so are several baked into
# the RoboDojo checkout. Passing local overrides ad hoc on a command line
# works once and then rots: the next person copies a half-remembered command,
# or worse, edits the AIDI submit configs to make a local run work and the
# next cluster submission fails. The submit configs under
# common/aidi_submit_config/ are for AIDI and must stay untouched; everything
# local lives here.
#
#   ./run_robodojo_eval_local.sh <gpu> <model_dir> <processor> [task[,task...]]
#
# Example (the random-init memoryvla package, one episode of swap_T):
#   ./run_robodojo_eval_local.sh 6 \
#     /horizon-bucket/.../holobrain_v10_mvla_randominit_smoke/package/ \
#     robodojo_arx_x5a_processor swap_T
#
# The six pod assumptions this file exists to override, each found by hitting
# it:
#
#   1. --robodojo_root  /opt/robodojo                 -> the local checkout
#   2. --conda_root     /opt/conda                    -> ~/miniconda3
#   3. --policy_env     /opt/holobrain_policy_env     -> the holobrain_internal
#      env by name. The image ships a dedicated policy env; the dev box has no
#      equivalent, and holobrain_internal is the only env with both torch and
#      robo_orchard_lab.
#   4. --kit_root       /job_data/.cache/isaacsim-kit -> JFS. Not overriding
#      this is a PermissionError on /job_data before anything else runs.
#   5. msgpack_numpy is missing from holobrain_internal (msgpack and
#      websockets are present). Injected by PYTHONPATH from a shim directory
#      rather than installed, because this port holds the host env at zero
#      changes and every equivalence baseline was measured inside it.
#   6. utils/update_embodiment_config_path.py has to have been run in the
#      RoboDojo checkout, or the robot URDF still resolves to
#      /running_package/code_package/. The AIDI cmd does this too; locally
#      nothing does it for you. This script runs it.
#
# Assets must NOT resolve to the bucket. It is 41 GB over 13,014 files, and
# reading it from there makes scene setup take minutes -- long enough that the
# websocket keepalive fires mid-load and the run enters a reconnect loop
# (measured: 16 scene loads and 11 keepalive timeouts in 17 minutes, zero
# episodes). Until 2026-08-07 --assets_dir did not fix it either, for a
# duller reason than the USD/MDL story that used to be written here: this
# checkout had no reader for ROBODOJO_ASSETS_DIR at all, nor for
# ROBODOJO_ENV_CONFIG_DIR or ROBODOJO_EVAL_RESULT_DIR, while the AIDI image
# honoured all three. Those three indirections have since been ported back
# into the RoboDojo repo and robodojo_eval.py now probes for them before
# Isaac Sim starts. The symlink below stays anyway: it costs nothing, and
# whether USD/MDL internal references also follow ASSETS_PATH has not been
# retested. It points at a JFS mirror:
#     /jfs-public/users/kun01.wu/xiaomi_robodojo/Assets_local
# verified against the bucket original with tree_verify.py
# (RESULT EQUIVALENT, 13,014 files hashed, 41,270,836,679 bytes). The original
# target is recorded next to the mirror in Assets_symlink_original.txt.
# No `set -u`: holobrain_internal's conda activate.d references
# NVCC_PREPEND_FLAGS unset, so -u kills the script at `conda activate`.
# r4_gear.sh carries the same note, for the same reason, from the same bite.
set -eo pipefail

GPU=${1:?gpu index, or a comma list like 4,5}
MODEL_DIR=${2:?deploy package dir}
PROCESSOR=${3:?processor name, e.g. robodojo_arx_x5a_processor}
TASKS=${4:-swap_T}
# Leave EVAL_NUM alone for a scoring run. summarize_result.py leaves the
# whole cell blank when a task has fewer than its native episode count
# (50 for a standalone task), so a short run is not a small score -- it is
# no score. 50 is what robodojo_eval.py calls STANDALONE_EPISODES.
EVAL_NUM=${EVAL_NUM:-50}
PROCS_PER_GPU=${PROCS_PER_GPU:-1}
# arx_x5 ships intrinsic_matrix/extrinsic_matrix false and robodojo_eval.py
# patches them true into --env_config_dir. That patch was read on the
# cluster and ignored here until the port above, which is why local arx_x5
# runs died on `camera ... is missing intrinsic_matrix` while the identical
# AIDI config was fine. Both honour it now; arx_x5_holobrain still works and
# is the only config that ever did locally.
ENV_CONFIG=${ENV_CONFIG:-arx_x5}
VALID_ACTION_STEP=${VALID_ACTION_STEP:-}

REPO=${REPO:-$HOME/git_repo/robo_orchard_lab}
RD_ROOT=${RD_ROOT:-$HOME/git_repo/RoboDojo}
COMMON=$REPO/projects/holobrain_internal/common

# The benchmark owns the eval landing, so its env is the one that counts.
# Both self-checks: env_selfcheck watches where caches land, bench_selfcheck
# watches where products land and whether that layer supports the writes the
# evaluator actually makes. They do not cover the same failures.
source "$RD_ROOT/robodojo_env.sh" >/dev/null
[ -n "$BENCH_LIVE" ] || { echo "BENCH_LIVE empty -- robodojo_env.sh not sourced"; exit 1; }
env_selfcheck   || { echo "env_selfcheck FAILED";   exit 1; }
bench_selfcheck || { echo "bench_selfcheck FAILED"; exit 1; }

RUN_TAG=${RUN_TAG:-run_$(date +%Y%m%d_%H%M%S)}
# Scratch is benchmark runtime state, so it lives under the benchmark tmp,
# not under the model project.
OUT=$TMPDIR
[ -n "$OUT" ] || { echo "TMPDIR empty -- env not sourced"; exit 1; }
RUN=$OUT/localeval/$RUN_TAG
mkdir -p "$RUN" "$OUT/localeval/kit/$RUN_TAG" "$OUT/localeval/pyshim"
OUT=$OUT/localeval

# (5) one pure-python file, copied not installed
if [ ! -f "$OUT/pyshim/msgpack_numpy.py" ]; then
  install -m 644 \
    "$HOME/miniconda3/envs/RoboDojo/lib/python3.11/site-packages/msgpack_numpy.py" \
    "$OUT/pyshim/"
fi

# (6) rewrite embodiment paths for this machine
source "$HOME/miniconda3/etc/profile.d/conda.sh"
( cd "$RD_ROOT" && conda run -n RoboDojo python utils/update_embodiment_config_path.py >/dev/null )

ulimit -n 65536 || true
conda activate holobrain_internal
cd "$COMMON"

export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
export CUDA_VISIBLE_DEVICES="$GPU"
# Deliberately NOT setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
# It helps when the card is shared and fragmented, but expandable segments
# and CUDA graph capture do not get along, and curobo's IK solver captures:
# with it set, the run dies in solve_pose with
# CUDA_ERROR_STREAM_CAPTURE_INVALIDATED. Set ALLOC_CONF explicitly if you
# knowingly want it.
if [ -n "$ALLOC_CONF" ]; then export PYTORCH_CUDA_ALLOC_CONF="$ALLOC_CONF"; fi
export PYTHONPATH=$OUT/pyshim

ARGS=(
  --policy_source "$COMMON/holobrain_robodojo_policy"
  --model_dir "$MODEL_DIR"
  --model_processor "$PROCESSOR"
  --robodojo_root "$RD_ROOT"                       # (1)
  --conda_root "$HOME/miniconda3"                  # (2)
  --policy_env holobrain_internal                  # (3)
  --kit_root "$OUT/kit/$RUN_TAG"                            # (4)
  --assets_dir "$ROBODOJO_ASSETS_DIR"
  --env_config_dir "$RUN/envcfg"
  --eval_result_dir "$BENCH_LIVE"
  --run_tag "$RUN_TAG"
  --vlm_ckpt_dir "$(readlink -f "$COMMON/ckpt")"
  --urdf_dir "$(readlink -f "$COMMON/urdf")"
  --eval_num "$EVAL_NUM"
  --processes_per_gpu "$PROCS_PER_GPU"
  --env_config "$ENV_CONFIG"
  --tasks "$TASKS"
)
# if, not `[ ] &&`: under set -e a false test as the last command is fatal
if [ -n "$VALID_ACTION_STEP" ]; then
  ARGS+=(--valid_action_step "$VALID_ACTION_STEP")
fi

echo "run dir: $RUN"
echo "gpu $GPU | tasks $TASKS | eval_num $EVAL_NUM | vas ${VALID_ACTION_STEP:-default}"
python3 robodojo_eval.py "${ARGS[@]}" 2>&1 | tee "$RUN/run.log"
