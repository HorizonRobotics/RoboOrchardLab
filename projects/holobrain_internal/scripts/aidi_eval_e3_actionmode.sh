#!/usr/bin/env bash
# E3 -- what does per-step forwarding actually cost, and does it take effect?
#
# Sizing before scheduling. Per-step forwarding replaces 25 policy forwards per
# episode with 800, and the only estimate available is a back-of-envelope
# 5-6x on wall clock derived from E2c's n1 (8 episodes in 35 min). Booking
# 50-episode cells on that number is a guess; two episodes per mode measures
# it. A 5090 slot costs less than a wrong schedule.
#
# All three stages are num_envs=1 on the mem package, so they are directly
# comparable to the cells already measured (5/50 fixed, 9/50 old numbering).
# num_envs > 1 is deliberately excluded: it scores 0.0 everywhere and its
# cause is still open, so mixing it in here would confound the one thing this
# run is for.
#
# eval_num=2 rather than 1: the `policy reset` line carrying every counter is
# emitted when the NEXT episode starts, so a single episode reports nothing --
# and two episodes also exercise the episode-boundary clearing path, which one
# never does.
#
# No `set -e`: chunk is the reference the other two are compared against and
# must survive them failing.
set -uo pipefail

RUN_DIR="${1:?usage: aidi_eval_e3_actionmode.sh <bucket run dir>}"
WORK="${WORKING_PATH:?WORKING_PATH unset}"

PKG=/horizon-bucket/robot_lab/users/kun01.wu/robo_orchard_lab/ckpts/memoryvla_eval_pkgs/100k_memory6_mem
TASK=cover_blocks
EVAL_NUM=2
OUT="${E3_OUT:-/job_data/eval_out}"
RD="${E3_RD:-/job_data/robodojo}"
LOG="$RUN_DIR/logs/stages.txt"
CAP="$RUN_DIR/logs/pod_capability.txt"
DRY="${E3_DRYRUN:-0}"

mkdir -p "$RUN_DIR/logs" "$OUT"
say() { echo "$*" | tee -a "$LOG"; }

{
  echo "=== pod $(date -u +%FT%TZ) uid=$(id -u) host=$(hostname) ==="
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader \
    2>/dev/null || echo "(no nvidia-smi)"
} | tee -a "$CAP"

if [ "$DRY" = "1" ]; then
  say "### DRY RUN -- skipping package open, tree patch and eval"
else
  /usr/bin/python3 -c "
open('$PKG/model.config.json', 'rb').read(64); print('OPEN OK pkg')
" | tee -a "$CAP" || { echo "FATAL pkg unreadable" | tee -a "$CAP"; exit 90; }

  say "### patching RoboDojo tree $(date -u +%FT%TZ)"
  bash "${WORK}/robodojo_pod_tree.sh" "$RD" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  [ "$rc" = "0" ] || { say "FATAL robodojo_pod_tree.sh rc=$rc"; exit "$rc"; }
fi

RC_ALL=0

run_stage() {  # name, action_mode
  local name="$1"
  local mode="$2"
  local rc=0
  local t0
  t0=$(date -u +%s)
  say "### stage $name start $(date -u +%FT%TZ) action_mode=$mode"
  rm -rf "$OUT"; mkdir -p "$OUT"

  if [ "$DRY" = "1" ]; then
    mkdir -p "$OUT/w0"
    /usr/bin/python3 - "$OUT/w0" "$mode" <<'FAKE'
import json, os, sys
d, mode = sys.argv[1], sys.argv[2]
# E3_DRYRUN_BREAK fabricates the failure this run exists to catch: the env var
# set, the log saying so, and the forward count still that of chunk mode --
# i.e. the mode was announced but never took effect.
broke = os.environ.get("E3_DRYRUN_BREAK") == "1"
fwd = 25 if (mode == "chunk" or broke) else 800
line = {"eval_episode": 1, "eval_forwards": fwd,
        "eval_history_reads": fwd - 1,
        "bank_keys": {"per_mem_bank": ["eval-env0-ep1"]},
        "action_mode": mode, "env_step": 800,
        "env_step_by_env": {"0": 800},
        "action_path_by_env": {"0": 1234.5},
        "action_jump_by_env": {"0": 12.3}}
open(d + "/eval.log", "w").write(f"INFO policy reset: {line}\n")
json.dump({"details": {str(i): {"layout_id": i, "score": 0.05}
                       for i in range(2)}}, open(d + "/_result.json", "w"))
FAKE
  else
  # The mode is chosen here and nowhere else. HOLOBRAIN_STEP_INDEX_MODE is
  # left unset on purpose: perstep/ensemble already force stride 1, and in
  # chunk mode unset means the corrected numbering, which is what the cells
  # this stage is compared against used.
  HOLOBRAIN_ACTION_MODE="$mode" \
  /usr/bin/python3 robodojo_eval.py \
    --policy_source "${WORK}/holobrain_robodojo_policy" \
    --model_dir "$PKG" \
    --model_processor robodojo_arx_x5a_processor \
    --env_config arx_x5 \
    --robodojo_root "$RD" \
    --eval_num "$EVAL_NUM" \
    --processes_per_gpu 1 \
    --num_envs 1 \
    --seed 0 \
    --valid_action_step 32 \
    --vlm_ckpt_dir /horizon-bucket/robot_lab/users/xuewu.lin/ckpt \
    --urdf_dir /horizon-bucket/robot_lab2/datasets/all_data/urdf/urdf_v20260711 \
    --eval_result_dir "$OUT" \
    --run_tag "$name" \
    --tasks "$TASK" || rc=$?
  fi
  RC_ALL=$((RC_ALL + rc))

  local secs=$(( $(date -u +%s) - t0 ))

  /usr/bin/python3 - "$OUT" "$name" "$mode" "$secs" "$EVAL_NUM" <<'PY' | tee -a "$LOG"
import ast, json, pathlib, sys
out, name, mode, secs, eval_num = sys.argv[1:6]
secs, eval_num = int(secs), int(eval_num)

modes, forwards, paths = set(), [], []
for log in pathlib.Path(out).rglob("*.log"):
    for line in log.read_text(errors="replace").splitlines():
        if "policy reset" not in line or "{" not in line:
            continue
        try:
            d = ast.literal_eval(line[line.index("{"):])
        except Exception:
            continue
        if d.get("action_mode"):
            modes.add(d["action_mode"])
        if d.get("eval_forwards"):
            forwards.append(d["eval_forwards"])
        for v in (d.get("action_path_by_env") or {}).values():
            paths.append(v)

scores = {}
for f in pathlib.Path(out).rglob("_result.json"):
    d = json.load(open(f))
    for k, v in (d.get("details") or {}).items():
        scores[str(v.get("layout_id", k))] = v.get("score")

per_ep = secs / eval_num if eval_num else 0
print(f"[prov {name}] mode_seen={sorted(modes) or None} "
      f"forwards_per_episode={forwards or None} "
      f"action_path={paths or None} "
      f"wall={secs}s ({per_ep:.0f}s/episode)")
print(f"[prov {name}] scores={json.dumps(scores, sort_keys=True)}")

ok = True
if not forwards:
    print(f"[prov {name}] FAIL no episode reported")
    ok = False
# Assert on the product, not on the variable that was exported. Setting
# HOLOBRAIN_ACTION_MODE proves nothing about whether the policy read it --
# three patches this week set something the code never consulted.
if modes != {mode}:
    print(f"[prov {name}] FAIL policy reported mode {sorted(modes) or None}, "
          f"asked for {mode!r}")
    ok = False
want_many = mode != "chunk"
if forwards:
    lo, hi = min(forwards), max(forwards)
    if want_many and lo < 600:
        print(f"[prov {name}] FAIL only {lo} forward(s) per episode. Per-step "
              "forwarding must run one per frame (~800); the mode was "
              "announced but the chunk is still being executed open loop.")
        ok = False
    if not want_many and hi > 64:
        print(f"[prov {name}] FAIL {hi} forwards per episode in chunk mode -- "
              "this stage is supposed to be the unchanged reference")
        ok = False
if paths and max(paths) <= 0:
    print(f"[prov {name}] FAIL the arm was never commanded to move")
    ok = False
print(f"[prov {name}] {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 3)
PY
  local prov_rc=${PIPESTATUS[0]}
  if [ "$prov_rc" != "0" ]; then
    say "### stage $name ASSERTION FAILED (parser rc=$prov_rc)"
    RC_ALL=$((RC_ALL + prov_rc))
  fi

  cp -r "$OUT/." "$RUN_DIR/" 2>&1 | tail -3 || true
  say "### stage $name done rc=$rc wall=${secs}s $(date -u +%FT%TZ)"
}

run_stage t_chunk   chunk
run_stage t_perstep perstep
run_stage t_ens     ensemble

say "ALL STAGES DONE rc_sum=$RC_ALL $(date -u +%FT%TZ)"
exit "$RC_ALL"
