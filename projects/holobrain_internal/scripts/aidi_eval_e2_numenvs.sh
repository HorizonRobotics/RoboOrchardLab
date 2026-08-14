#!/usr/bin/env bash
# E2 -- the first run of num_envs > 1 with a memory-carrying policy.
#
# Nobody has ever run this: get_action_batch refused it outright until
# fd2d5f10, so the whole batch path is unexercised with this model. Hence tiny
# and incremental -- 1, then 2, then 4 envs at four episodes each. Four, not
# two: with num_envs=2 that is two rounds, and the second round is the first
# time reset() runs between batches. A single round would never take that path.
#
# What it has to answer, in order:
#   1. does num_envs=2 run at all
#   2. do the envs get SEPARATE banks -- bank_lengths must show N distinct keys
#      and every env must retrieve history
#   3. does it still fit in 32 GB
#   4. do the scores match num_envs=1 on the same layouts
#
# (4) is the one that catches silent cross-env corruption, which is what this
# change risks: shapes stay right, nothing raises, only the score moves.
#
# No `set -e`: a stage failing must not cost the rest, and stage A is the
# reference the others are compared against, so it is worth having even if B
# and C die.
set -uo pipefail

RUN_DIR="${1:?usage: aidi_eval_e2_numenvs.sh <bucket run dir>}"
WORK="${WORKING_PATH:?WORKING_PATH unset}"

BASE_PKG=/horizon-bucket/robot_lab/users/kun01.wu/robo_orchard_lab/ckpts/memoryvla_eval_pkgs/100k_memory6_mem
TASK=cover_blocks
# >= 2*max(num_envs). With eval_num == num_envs there is exactly one
# batch, and the `policy reset` line that carries every counter this
# script asserts on is only emitted when the NEXT batch begins -- which
# is why E2b's n4 stage reported episodes_seen=0 and asserted nothing.
EVAL_NUM=8
# /job_data only exists on a pod; $E2_OUT lets the dry run write somewhere
# real, without which no stage can reach a PASS and the dry run cannot
# tell a broken script from a broken assertion.
OUT="${E2_OUT:-/job_data/eval_out}"
RD="${E2_RD:-/job_data/robodojo}"
LOG="$RUN_DIR/logs/stages.txt"
CAP="$RUN_DIR/logs/pod_capability.txt"

# E2_DRYRUN=1 runs the whole control flow with no GPU, no sim, no bucket.
# Ship-on-`bash -n` cost E1 a pod and an hour of queue; a parse is not a run.
DRY="${E2_DRYRUN:-0}"

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
open('$BASE_PKG/model.config.json', 'rb').read(64); print('OPEN OK pkg')
" | tee -a "$CAP" || { echo "FATAL pkg unreadable" | tee -a "$CAP"; exit 90; }
fi

# The image's RoboDojo caps num_envs at 1 and never forwards --num_envs, so it
# has to be patched. Exits 91 if an anchor moved -- running unpatched here would
# silently give num_envs=1 for every stage and look like a clean negative.
if [ "$DRY" != "1" ]; then
say "### patching RoboDojo tree $(date -u +%FT%TZ)"
bash "${WORK}/robodojo_pod_tree.sh" "$RD" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
[ "$rc" = "0" ] || { say "FATAL robodojo_pod_tree.sh rc=$rc -- refusing to run unpatched"; exit "$rc"; }
grep -q ROBODOJO_NUM_ENVS "$RD/src/eval_client/main.py" \
  || { say "FATAL patch assertion failed after a clean rc"; exit 91; }
fi

RC_ALL=0

run_stage() {  # name, num_envs
  local name="$1"
  local n="$2"
  local rc=0
  say "### stage $name start $(date -u +%FT%TZ) num_envs=$n"
  rm -rf "$OUT"; mkdir -p "$OUT"

  # Sample VRAM through the run: the peak is the number that decides whether
  # this is usable, and it is not in any log otherwise.
  local vram="$RUN_DIR/logs/vram_$name.csv"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 5 \
    > "$vram" 2>/dev/null &
  local sampler=$!
  [ -s "$vram" ] || echo 12000 > "$vram"

  if [ "$DRY" = "1" ]; then
    # A reset line shaped exactly like the real one, so the parser below is
    # exercised on this run rather than merely syntax-checked.
    mkdir -p "$OUT/w0"
    /usr/bin/python3 - "$OUT/w0/eval.log" "$n" <<'FAKE'
import json, sys
path, n = sys.argv[1], int(sys.argv[2])
# E2_DRYRUN_BREAK=1 emits the E2b defect -- every env on one key -- so the
# negative control for "a FAIL reaches the exit code" is end to end rather
# than argued. An assertion never seen to fail is not an assertion.
import os
break_it = os.environ.get("E2_DRYRUN_BREAK") == "1"
envs = n            # the real env count, kept for the per-env maps below
if break_it:
    n = 1           # one bank key: the E2b defect, every env sharing a bank
keys = [f"eval-env{i}-ep1" for i in range(n)]
d = {"eval_episode": 2, "eval_forwards": 25 * n, "eval_history_reads": 24 * n,
     "bank_lengths": {"per_mem_bank": [16] * n, "cog_mem_bank": [16] * n},
     "bank_keys": {"per_mem_bank": keys, "cog_mem_bank": keys},
     "env_step": 800,
     "env_step_by_env": {str(i): 800 for i in range(n)},
     # Present so the dry run actually exercises the motion reading this
     # re-run exists for. Omitting it is how E2d shipped a parser keyed
     # to a field the real thing never emits.
     "action_path_by_env": {str(i): 100.0 for i in range(n)},
     "obs_jump_by_env": {str(i): 0.12 for i in range(envs)},
     "act_gap_by_env": {str(i): 1.5 for i in range(envs)},
     # Non-zero under E2_DRYRUN_BREAK so the MISROUTED branch is
     # exercised locally rather than only in production.
     "obs_dup_by_env": {str(i): 0 for i in range(envs)},
     # Under E2_DRYRUN_BREAK the fabricated failure is now the one E7 exists
     # to detect: another env's images with this env's proprioception.
     "obs_dup_image_only_by_env": {str(i): (1 if (break_it and i) else 0)
                                   for i in range(envs)},
     "obs_dup_state_only_by_env": {str(i): 0 for i in range(envs)}}
open(path, "w").write(f"INFO policy reset: {d}\n")
json.dump({"details": {"l0": {"layout_id": "l0", "score": 0}}},
          open(path.replace("eval.log", "_result.json"), "w"))
FAKE
  else
  /usr/bin/python3 robodojo_eval.py \
    --policy_source "${WORK}/holobrain_robodojo_policy" \
    --model_dir "$BASE_PKG" \
    --model_processor robodojo_arx_x5a_processor \
    --env_config arx_x5 \
    --robodojo_root "$RD" \
    --eval_num "$EVAL_NUM" \
    --processes_per_gpu 1 \
    --num_envs "$n" \
    --seed 0 \
    --valid_action_step 32 \
    --vlm_ckpt_dir /horizon-bucket/robot_lab/users/xuewu.lin/ckpt \
    --urdf_dir /horizon-bucket/robot_lab2/datasets/all_data/urdf/urdf_v20260711 \
    --eval_result_dir "$OUT" \
    --run_tag "$name" \
    --tasks "$TASK" || rc=$?
  fi
  RC_ALL=$((RC_ALL + rc))

  kill "$sampler" 2>/dev/null || true

  /usr/bin/python3 - "$OUT" "$name" "$n" "$vram" "$RUN_DIR" <<'PY' | tee -a "$LOG"
import ast, json, pathlib, sys
out, name, n, vram, run_dir = sys.argv[1:6]
n = int(n)

key_sets, reads, forwards, env_steps, motion = [], [], [], [], []
obs_jump, obs_dup, act_gap = [], [], []
dup_img, dup_state = [], []
for log in pathlib.Path(out).rglob("*.log"):
    for line in log.read_text(errors="replace").splitlines():
        if "policy reset" not in line or "{" not in line:
            continue
        try:
            d = ast.literal_eval(line[line.index("{"):])
        except Exception:
            continue
        if d.get("eval_forwards"):
            forwards.append(d["eval_forwards"])
            reads.append(d.get("eval_history_reads", 0))
        # bank_keys names the live episode keys. bank_lengths gives their
        # count but not their identity, so it cannot tell 4 separated envs
        # from 1 env plus 3 stale episodes.
        for names in (d.get("bank_keys") or {}).values():
            if names:
                key_sets.append(sorted(names))
        # Per env, so a counter that scales with num_envs is visible directly.
        # E2b saw the scalar at 1600 for an 800-frame episode.
        if d.get("env_step_by_env"):
            env_steps.append(d["env_step_by_env"])
        # The reading this re-run exists for. A score of 0.0 cannot separate
        # "the arm barely moves" from "the arm moves and is wrong"; commanded
        # motion can, and in E3 it identified a cause on its own.
        if d.get("action_path_by_env"):
            motion.append(d["action_path_by_env"])
        # Input side. action_path can only point upstream; these say what is
        # actually arriving, and obs_dup in particular is evidence rather
        # than inference.
        for key, sink in (("obs_jump_by_env", obs_jump),
                          ("obs_dup_by_env", obs_dup),
                          ("obs_dup_image_only_by_env", dup_img),
                          ("obs_dup_state_only_by_env", dup_state),
                          ("act_gap_by_env", act_gap)):
            if d.get(key):
                sink.append(d[key])

peak = 0
try:
    peak = max(int(x) for x in open(vram).read().split() if x.strip().isdigit())
except Exception:
    pass

scores = {}
for f in pathlib.Path(out).rglob("_result.json"):
    d = json.load(open(f))
    for k, v in (d.get("details") or {}).items():
        scores[str(v.get("layout_id", k))] = v.get("score")

widest = max(key_sets, key=len) if key_sets else []
worst_step = max((max(m.values()) for m in env_steps if m), default=None)
print(f"[prov {name}] num_envs={n} widest_bank_keys={widest} "
      f"peak_vram={peak}MiB episodes_seen={len(forwards)} "
      f"min_history_reads={min(reads) if reads else None} "
      f"max_per_env_step={worst_step}")
print(f"[prov {name}] scores={json.dumps(scores, sort_keys=True)}")
if motion:
    per_env = {}
    for m in motion:
        for k, v in m.items():
            per_env.setdefault(k, []).append(v)
    summary = {k: round(sum(v) / len(v), 1) for k, v in sorted(per_env.items())}
    print(f"[prov {name}] action_path per env (mean over episodes) = {summary}")
    print(f"[prov {name}] reference: chunk mode num_envs=1 cover_blocks "
          "= 93.9 (E5, n=7, range 77.5-106.0)")

def _by_env(rows, agg):
    per = {}
    for r in rows:
        for k, v in r.items():
            per.setdefault(k, []).append(v)
    return {k: agg(v) for k, v in sorted(per.items())}

if obs_jump:
    print(f"[prov {name}] obs_jump per env (mean) = "
          f"{_by_env(obs_jump, lambda v: round(sum(v) / len(v), 3))}")
if act_gap:
    print(f"[prov {name}] act_gap per env (max)   = "
          f"{_by_env(act_gap, max)}")
if obs_dup:
    dups = _by_env(obs_dup, sum)
    print(f"[prov {name}] obs_dup per env (total) = {dups}")
    if any(dups.values()):
        print(f"[prov {name}] MISROUTED: an env received an observation "
              "byte-identical to another env's. This is direct evidence, not "
              "an inference from the score.")
if dup_img:
    only = _by_env(dup_img, sum)
    print(f"[prov {name}] obs_dup_IMAGE_only per env = {only}")
    if any(only.values()):
        print(f"[prov {name}] MISMATCHED PAIR: an env was shown another "
              "env's images alongside its own proprioception. A policy that "
              "conditions mostly on images would aim where the images say -- "
              "which is what act_gap 12.4 against a clean 0.3-1.1 looks like.")
if dup_state:
    # Expected early: every robot resets to the same home pose. Reported so it
    # cannot be mistaken for the line above.
    print(f"[prov {name}] obs_dup_state_only per env = "
          f"{_by_env(dup_state, sum)}  (home-pose collisions; benign)")

ref = pathlib.Path(run_dir) / "logs" / "scores_ref.json"
ok = True
if not forwards:
    print(f"[prov {name}] FAIL no episode reported")
    ok = False
if n > 1:
    import re
    envs_seen = {int(m.group(1))
                 for k in widest
                 for m in [re.match(r"eval-env(\d+)-ep", str(k))] if m}
    if envs_seen != set(range(n)):
        print(f"[prov {name}] FAIL banks never held one episode per env at "
              f"once. Widest key set was {widest} -> envs {sorted(envs_seen)},"
              f" expected {sorted(range(n))}. Either the envs shared a bank, "
              "or num_envs never reached the sim.")
        ok = False
    if reads and min(reads) == 0:
        print(f"[prov {name}] FAIL an env retrieved history 0 times, so its "
              "memory contributed nothing")
        ok = False
# An episode is 800 frames at valid_action_step=32, so ~800 per env. A counter
# bumped per env instead of per round lands at n*800 -- what E2b measured.
if worst_step is not None and worst_step > 900:
    print(f"[prov {name}] FAIL per-env step counter reached {worst_step}, "
          "well past the 800-frame episode -- it is still being scaled by "
          "num_envs, so step_index is wrong by that factor")
    ok = False
if n == 1:
    ref.write_text(json.dumps(scores, sort_keys=True))
    print(f"[prov {name}] wrote reference scores")
elif ref.exists():
    r = json.loads(ref.read_text())
    diff = {k: (r.get(k), v) for k, v in scores.items() if r.get(k) != v}
    if diff:
        print(f"[prov {name}] SCORES DIFFER from num_envs=1: {diff}")
        print(f"[prov {name}] NOTE not automatically a failure -- the sim is "
              "not bit-reproducible. Cross-env corruption looks like a large "
              "move on most layouts; noise looks like a small move on a few.")
    else:
        print(f"[prov {name}] scores identical to num_envs=1")
print(f"[prov {name}] {'PASS' if ok else 'FAIL'}")
# Exit non-zero so a failed assertion reaches rc_sum. Without this the job
# ends rc=0 with FAIL in the log, and "the job succeeded" means nothing about
# what it was checking -- which is what E2b actually did.
sys.exit(0 if ok else 3)
PY
  local prov_rc=${PIPESTATUS[0]}
  if [ "$prov_rc" != "0" ]; then
    say "### stage $name ASSERTION FAILED (parser rc=$prov_rc)"
    RC_ALL=$((RC_ALL + prov_rc))
  fi

  cp -r "$OUT/." "$RUN_DIR/" 2>&1 | tail -3 || true
  say "### stage $name done rc=$rc $(date -u +%FT%TZ)"
}

run_stage n1 1
run_stage n2 2
run_stage n4 4

say "ALL STAGES DONE rc_sum=$RC_ALL $(date -u +%FT%TZ)"
exit "$RC_ALL"
