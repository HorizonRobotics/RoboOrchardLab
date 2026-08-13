#!/usr/bin/env bash
# E4 -- the ensemble scheme at full strength, so its score can be compared.
#
# E3 measured what E2c could not: per-step forwarding works mechanically (800
# forwards per episode, one per frame, read off the product rather than the
# exported variable) and costs 3.2x wall clock, not the 5-6x estimated.
#
# It also produced the finding this run follows up. Commanded motion, the
# observable added for the num_envs mystery:
#
#     chunk     action_path = 103.75
#     perstep   action_path =  44.68   (43%)
#     ensemble  action_path =  89.23   (86%)
#
# So executing only a[0] of each chunk moves the arm less than half as far --
# the first action of a chunk is the timid one, and chunk mode escapes that by
# executing the larger later actions. ACT's ensemble restores 86% of it,
# exactly as predicted before the stage ran, by mixing in the offset-k entries
# of predictions made up to 31 frames earlier (its weights favour the older
# ones).
#
# What E3 could NOT answer is the score: both new modes read 0.0 on both
# layouts, but two layouts decide nothing when one layout has been measured at
# 0.05 and 1.0 on two runs of an identical config. Pooled, 0/4 vs chunk's 2/2
# is p=0.067 -- suggestive, underpowered, not a result.
#
# Hence 50 episodes, the same count as every cell it must be compared with:
#
#     cover_blocks, seed 0, mem package, num_envs=1
#       old numbering, chunk   9/50
#       fixed numbering, chunk 5/50
#       ensemble               <- this run
#
# Parallelism in robodojo_eval.py is by task, not by episode
# (worker_count = min(len(task_names), gpus * processes_per_gpu)), so a
# single-task cell is 7.6 h on one worker whatever the GPU count. Running both
# tasks costs the same wall clock and yields the conveyor cell for free.
#
# num_envs stays 1: it scores 0.0 everywhere and its cause is open, so mixing
# it in would confound the one thing this run is for.
set -uo pipefail

RUN_DIR="${1:?usage: aidi_eval_e4_ensemble.sh <bucket run dir>}"
WORK="${WORKING_PATH:?WORKING_PATH unset}"

PKG=/horizon-bucket/robot_lab/users/kun01.wu/robo_orchard_lab/ckpts/memoryvla_eval_pkgs/100k_memory6_mem
TASKS=cover_blocks,match_and_pick_from_conveyor
EVAL_NUM=50
SEED=0
OUT="${E4_OUT:-/job_data/eval_out}"
RD="${E4_RD:-/job_data/robodojo}"
LOG="$RUN_DIR/logs/stages.txt"
CAP="$RUN_DIR/logs/pod_capability.txt"
DRY="${E4_DRYRUN:-0}"

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
t0=$(date -u +%s)
say "### stage ens50 start $(date -u +%FT%TZ) mode=ensemble seed=$SEED tasks=$TASKS"

if [ "$DRY" = "1" ]; then
  mkdir -p "$OUT/w0"
  /usr/bin/python3 - "$OUT/w0" <<'FAKE'
import json, os, sys
d = sys.argv[1]
# E4_DRYRUN_BREAK reproduces the failure worth catching: the mode announced
# but the chunk still executed open loop, which would look like a clean
# ensemble cell whose number is actually chunk mode's.
broke = os.environ.get("E4_DRYRUN_BREAK") == "1"
line = {"eval_episode": 1, "eval_forwards": 25 if broke else 800,
        "action_mode": "ensemble", "env_step": 800,
        "action_path_by_env": {"0": 89.2}}
open(d + "/eval.log", "w").write(f"INFO policy reset: {line}\n")
json.dump({"details": {str(i): {"layout_id": i, "score": 0.05}
                       for i in range(50)}}, open(d + "/_result.json", "w"))
FAKE
else
  HOLOBRAIN_ACTION_MODE=ensemble \
  /usr/bin/python3 robodojo_eval.py \
    --policy_source "${WORK}/holobrain_robodojo_policy" \
    --model_dir "$PKG" \
    --model_processor robodojo_arx_x5a_processor \
    --env_config arx_x5 \
    --robodojo_root "$RD" \
    --eval_num "$EVAL_NUM" \
    --processes_per_gpu 1 \
    --num_envs 1 \
    --seed "$SEED" \
    --valid_action_step 32 \
    --vlm_ckpt_dir /horizon-bucket/robot_lab/users/xuewu.lin/ckpt \
    --urdf_dir /horizon-bucket/robot_lab2/datasets/all_data/urdf/urdf_v20260711 \
    --eval_result_dir "$OUT" \
    --run_tag ens50 \
    --tasks "$TASKS" || RC_ALL=$?
fi

secs=$(( $(date -u +%s) - t0 ))

/usr/bin/python3 - "$OUT" "$secs" <<'PY' | tee -a "$LOG"
import ast, json, pathlib, sys
out, secs = sys.argv[1], int(sys.argv[2])

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

per_task = {}
for f in sorted(pathlib.Path(out).rglob("_result.json")):
    det = json.load(open(f)).get("details") or {}
    task = f.parent.name.split("_ens50_")[-1] or f.parent.name
    wins = sum(1 for v in det.values() if v.get("score") == 1.0)
    part = sum(1 for v in det.values() if 0 < (v.get("score") or 0) < 1.0)
    per_task[task] = (wins, part, len(det))

lo = min(forwards) if forwards else None
print(f"[prov ens50] mode_seen={sorted(modes) or None} "
      f"forwards_per_episode min={lo} max={max(forwards) if forwards else None} "
      f"episodes_seen={len(forwards)} wall={secs}s")
if paths:
    print(f"[prov ens50] action_path n={len(paths)} "
          f"mean={sum(paths)/len(paths):.1f} min={min(paths):.1f} "
          f"max={max(paths):.1f}   (E3: chunk 103.8 / perstep 44.7 / ens 89.2)")
for task, (w, p, n) in sorted(per_task.items()):
    print(f"[prov ens50] {task}: {w}/{n} success (+{p} partial)")
print("[prov ens50] compare cover_blocks seed0: old numbering 9/50, "
      "fixed numbering 5/50, both chunk mode")

ok = True
if not forwards:
    print("[prov ens50] FAIL no episode reported")
    ok = False
# On the product, not the exported variable: three patches this week set
# something the code never read.
if modes != {"ensemble"}:
    print(f"[prov ens50] FAIL policy reported mode {sorted(modes) or None}")
    ok = False
if lo is not None and lo < 600:
    print(f"[prov ens50] FAIL an episode ran only {lo} forwards -- per-step "
          "forwarding did not take effect there, so its number is chunk "
          "mode's wearing an ensemble label")
    ok = False
if paths and max(paths) <= 0:
    print("[prov ens50] FAIL the arm was never commanded to move")
    ok = False
print(f"[prov ens50] {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 3)
PY
prov_rc=${PIPESTATUS[0]}
if [ "$prov_rc" != "0" ]; then
  say "### ASSERTION FAILED (parser rc=$prov_rc)"
  RC_ALL=$((RC_ALL + prov_rc))
fi

cp -r "$OUT/." "$RUN_DIR/" 2>&1 | tail -3 || true
say "ALL STAGES DONE rc_sum=$RC_ALL wall=${secs}s $(date -u +%FT%TZ)"
exit "$RC_ALL"
