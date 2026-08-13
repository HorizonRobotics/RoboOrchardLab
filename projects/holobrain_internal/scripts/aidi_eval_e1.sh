#!/usr/bin/env bash
# E1 -- the stride1 half of the step_index 2x2, plus two pure-config probes.
#
# See docs_analysis/memoryvla/INVESTIGATION-step_index-20260813.md. The
# prediction, recorded before this runs: fixing step_index does NOT raise
# success rate on a stride1-trained checkpoint and may lower it, because the
# fix moves the bank's timestep spacing from 1 (what stride1 training fed) to
# 32, and 13 of the 48 sinusoidal frequency dims alias at spacing 32.
#
# Four stages, each cover_blocks + conveyor at eval_num 50, run in sequence on
# one GPU (two tasks == two slots; _allocate_tasks hands one task per slot, so
# more GPUs would only idle):
#   s1_fix_s0   base pkg, step_index fixed          seed 0   vs known 9/50
#   s2_fix_s1   base pkg, step_index fixed          seed 1   vs known 14/50
#   a0_mem32    mem_length=32,     step_index old   seed 0   vs known 9/50
#   a1_fifo     consolidate=fifo,  step_index old   seed 0   vs known 9/50
#
# A0/A1 deliberately keep the OLD step_index numbering so exactly one variable
# separates them from the existing baseline.
#
# No `set -e`: one stage failing must not cost the other three. Each stage
# archives to the bucket immediately after itself, so a wall_time kill still
# leaves the finished stages behind -- /job_data vanishes with the job.
set -uo pipefail

RUN_DIR="${1:?usage: aidi_eval_e1.sh <bucket run dir>}"

# E1_DRYRUN=1 runs every branch with no GPU, no Isaac Sim and no bucket, so
# the control flow and the provenance assertions get exercised before a queue
# slot is spent on them. It was added after a `local` statement referring to
# its own earlier assignment killed the pod four seconds in -- bash -n is a
# parse, not a run.
DRY="${E1_DRYRUN:-0}"
WORK="${WORKING_PATH:-${PWD}}"
if [ "$DRY" = "0" ]; then
    : "${WORKING_PATH:?WORKING_PATH unset}"
fi

BASE_PKG=/horizon-bucket/robot_lab/users/kun01.wu/robo_orchard_lab/ckpts/memoryvla_eval_pkgs/100k_memory6_mem
if [ "$DRY" != "0" ]; then
  # The JFS copy the bucket one was built from -- same bytes, readable here.
  BASE_PKG=/jfs-public/users/kun01.wu/robo_orchard_lab/port/memoryvla/eval_pkgs/100k_memory6_mem
fi
TASKS='cover_blocks,match_and_pick_from_conveyor'
if [ "$DRY" = "0" ]; then
  PKGS=/job_data/pkgs
  OUT=/job_data/eval_out
else
  PKGS="$RUN_DIR/dry_pkgs"
  OUT="$RUN_DIR/dry_out"
fi
LOG="$RUN_DIR/logs/stages.txt"
CAP="$RUN_DIR/logs/pod_capability.txt"

mkdir -p "$RUN_DIR/logs" "$OUT" "$PKGS"
say() { echo "$*" | tee -a "$LOG"; }

# ---------------------------------------------------------------- preflight
{
  echo "=== pod $(date -u +%FT%TZ) uid=$(id -u) host=$(hostname) dry=$DRY ==="
  df -h /job_data 2>/dev/null | tail -1 || echo "(no /job_data)"
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader \
    2>/dev/null || echo "(no nvidia-smi)"
} | tee -a "$CAP"

# A real open(), not `test -f`: on JuiceFS-backed mounts ls/stat succeed while
# open() returns EACCES, because access control is done per requesting uid.
if [ "$DRY" = "0" ]; then
  /usr/bin/python3 -c "
for p in ('$BASE_PKG/model.config.json',
          '/horizon-bucket/robot_lab2/datasets/assets/robodojo_assets/Eval_Layout/RoboDojo/arx_x5/1/cover_blocks_0.json'):
    open(p, 'rb').read(64); print('OPEN OK', p)
" | tee -a "$CAP" || { echo "FATAL bucket not readable" | tee -a "$CAP"; exit 90; }
fi

touch "$OUT/.wtest" && echo JOB_DATA_WRITABLE | tee -a "$CAP" \
  || { echo FATAL_job_data_not_writable | tee -a "$CAP"; exit 90; }

# ------------------------------------------------------- config-variant pkgs
# Symlinks, not copies: model.safetensors is 2.4 GB and only the 15 KB
# model.config.json differs. The loader takes Path(model_dir) without
# resolve() (deploy_policy.py:248), so a directory of links is equivalent --
# but that equivalence is asserted below via A0's bank_lengths, not assumed.
#
# urdf and ckpt are linked to the BASE PACKAGE's own, not left for
# link_model_resource() to create from --urdf_dir/--vlm_ckpt_dir: that would
# be a second difference from the baseline, and the point of a pure-config
# probe is that there is exactly one.
make_variant() {  # name, python-expr mutating m
  # Split, not one `local`: `local a="$1" b="$PKGS/$a"` dies under set -u
  # with `a: unbound variable`, and bash -n does not see it.
  local name="$1"
  local expr="$2"
  local d="$PKGS/$name"
  local f
  rm -rf "$d"; mkdir -p "$d"
  for f in model.safetensors robodojo_arx_x5a_inference.config.json \
           robodojo_arx_x5a_processor.json urdf ckpt; do
    ln -s "$BASE_PKG/$f" "$d/$f"
  done
  /usr/bin/python3 -c "
import json
d = json.load(open('$BASE_PKG/model.config.json'))
m = d['memoryvla']
assert m['mem_length'] == 16 and m['consolidate_type'] == 'tome', m
$expr
json.dump(d, open('$d/model.config.json', 'w'), indent=2)
print('variant $name ->', {k: m[k] for k in ('mem_length', 'consolidate_type')})
" | tee -a "$LOG" || { say "FATAL could not build variant $name"; exit 91; }
}

make_variant a0_mem32 "m['mem_length'] = 32"
make_variant a1_fifo  "m['consolidate_type'] = 'fifo'"

# ------------------------------------------------------------------- stages
RC_ALL=0

run_stage() {  # name, pkg, step_index_mode, seed, expect_step, expect_bank
  local name="$1" pkg="$2" mode="$3" seed="$4" xstep="$5" xbank="$6" rc=0
  say "### stage $name start $(date -u +%FT%TZ) step_index=${mode:-fixed} seed=$seed"
  rm -rf "$OUT"; mkdir -p "$OUT"

  # Inherited all the way down: robodojo_eval.py:917 os.environ.copy() ->
  # :675 task_env -> conda run -> eval.sh -> setup_eval_policy_server.sh,
  # none of which scrubs the environment. Empty string != "forward", so an
  # empty value selects the fixed (default) numbering.
  export HOLOBRAIN_STEP_INDEX_MODE="$mode"

  if [ "$DRY" != "0" ]; then
    # Two episodes' worth of the line deploy_policy.reset() prints, with the
    # env_step this mode should produce: 25 forwards x 32 for the fix, a bare
    # forward count for the old numbering. bank_len follows mem_length.
    local dlog="$OUT/dry_worker/logs/cover_blocks.log"
    mkdir -p "$(dirname "$dlog")"
    local es=800 bl=16
    [ "$mode" = "forward" ] && es=25
    [ "$xbank" = "gt16" ] && bl=25
    {
      echo "INFO deploy_policy | policy reset: {'eval_episode': 0, 'eval_forwards': 0, 'eval_history_reads': 0, 'bank_lengths': {'per_mem_bank': [], 'cog_mem_bank': []}, 'env_step': 0}"
      echo "INFO deploy_policy | policy reset: {'eval_episode': 1, 'eval_forwards': 25, 'eval_history_reads': 24, 'bank_lengths': {'per_mem_bank': [$bl], 'cog_mem_bank': [$bl]}, 'env_step': $es}"
    } > "$dlog"
    rc=0
  else
  /usr/bin/python3 robodojo_eval.py \
    --policy_source "${WORK}/holobrain_robodojo_policy" \
    --model_dir "$pkg" \
    --model_processor robodojo_arx_x5a_processor \
    --env_config arx_x5 \
    --eval_num 50 \
    --processes_per_gpu 2 \
    --seed "$seed" \
    --valid_action_step 32 \
    --vlm_ckpt_dir /horizon-bucket/robot_lab/users/xuewu.lin/ckpt \
    --urdf_dir /horizon-bucket/robot_lab2/datasets/all_data/urdf/urdf_v20260711 \
    --eval_result_dir "$OUT" \
    --run_tag "$name" \
    --tasks "$TASKS" || rc=$?
  fi
  RC_ALL=$((RC_ALL + rc))

  # Provenance from the products, before archiving. A config file that was
  # edited but never read looks exactly like one that took effect, so assert
  # on what the policy itself reported per episode.
  /usr/bin/python3 - "$OUT" "$name" "$xstep" "$xbank" <<'PY' | tee -a "$LOG"
import ast, pathlib, sys
out, name, xstep, xbank = sys.argv[1:5]
steps, banks = [], []
for log in pathlib.Path(out).rglob("*.log"):
    for line in log.read_text(errors="replace").splitlines():
        if "policy reset" not in line or "{" not in line:
            continue
        try:
            d = ast.literal_eval(line[line.index("{"):])
        except Exception:
            continue
        if "env_step" in d:
            steps.append(int(d["env_step"]))
        for lens in (d.get("bank_lengths") or {}).values():
            banks.extend(int(x) for x in lens)
print(f"[prov {name}] reset lines={len(steps)} "
      f"env_step max={max(steps) if steps else None} "
      f"bank_len max={max(banks) if banks else None}")
ok = True
if not steps:
    print(f"[prov {name}] FAIL no `policy reset` line parsed")
    ok = False
elif xstep == "fixed" and max(steps) < 256:
    print(f"[prov {name}] FAIL env_step max {max(steps)} < 256 -- the fix did "
          "NOT take effect, still counting forwards")
    ok = False
elif xstep == "forward" and max(steps) > 64:
    print(f"[prov {name}] FAIL env_step max {max(steps)} > 64 -- old numbering "
          "was asked for but the fixed one ran")
    ok = False
if xbank != "any":
    if not banks:
        print(f"[prov {name}] FAIL no bank_lengths reported")
        ok = False
    elif xbank == "gt16" and max(banks) <= 16:
        print(f"[prov {name}] FAIL bank_len max {max(banks)} <= 16 -- "
              "mem_length=32 did NOT take effect, so the symlinked package "
              "was not read and a1_fifo is void too")
        ok = False
    elif xbank == "eq16" and max(banks) != 16:
        print(f"[prov {name}] FAIL bank_len max {max(banks)} != 16")
        ok = False
print(f"[prov {name}] {'PASS' if ok else 'FAIL'}")
PY

  if [ "$DRY" = "0" ]; then
    cp -r "$OUT/." "$RUN_DIR/" 2>&1 | tail -3 || true
  fi
  say "### stage $name done rc=$rc $(date -u +%FT%TZ)"
}

# The two step_index cells first: they are the deliverable, and a wall_time
# kill should not land on them. A0 still precedes A1 -- its bank_lengths is the
# canary for whether a symlinked package is read at all, and if that fails A1's
# number means nothing.
run_stage s1_fix_s0 "$BASE_PKG"      ""      0 fixed   eq16
run_stage s2_fix_s1 "$BASE_PKG"      ""      1 fixed   eq16
run_stage a0_mem32  "$PKGS/a0_mem32" forward 0 forward gt16
run_stage a1_fifo   "$PKGS/a1_fifo"  forward 0 forward eq16

say "ALL STAGES DONE rc_sum=$RC_ALL $(date -u +%FT%TZ)"
exit "$RC_ALL"
