#!/usr/bin/env bash
# E2d -- does num_envs > 1 break a policy that has no memory at all?
#
# E2c settled the two defects it was built for: with num_envs=2 and 4 the envs
# get separate banks (bank_keys shows one key per env) and each env's frame
# counter lands on 800 rather than num_envs*800. Both confirmed on hardware.
#
# And then every score went to zero. Same package, same seed, same 8 layouts:
#
#   layout      0     1     2     3     4     5     6     7
#   n1 (1 env)  0.05  0.05  0.05  0.05  0.0   0.05  0.05  1.00
#   n2 (2 env)  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
#   n4 (4 env)  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
#
# 0.05 is the lowest scoring segment, so under multi-env the policy does not
# reach even the first sub-goal anywhere, including the layout it completed
# outright at num_envs=1. Not noise, and not truncation: all 8 episodes ran,
# 25 forwards per env per batch, history retrieved 24 times per env.
#
# Two explanations remain and reading the code has picked the wrong one three
# times this week, so this is a control rather than an argument:
#
#   A. the memory retrieves across envs -- separate banks, mixed reads
#   B. something in RoboDojo's batch action/sim path is wrong for this policy
#      regardless of memory
#
# The baseline package has no memory at all -- step_index never enters the
# model input, so no bank is ever created (verified by the negative control in
# check_multi_env_isolation.py). If it collapses the same way, the memory is
# innocent and (B) is where to look. If it holds, (A).
#
# No `set -e`: b1 is the reference the comparison needs, so it must survive b2
# failing, and vice versa.
set -uo pipefail

RUN_DIR="${1:?usage: aidi_eval_e2d_basectl.sh <bucket run dir>}"
WORK="${WORKING_PATH:?WORKING_PATH unset}"

# The BASELINE package. This is the whole point of the run -- assert on it
# rather than trusting the path string, since a mem package here would produce
# a confidently wrong answer with nothing in the log to show for it.
BASE_PKG=/horizon-bucket/robot_lab/users/kun01.wu/robo_orchard_lab/ckpts/memoryvla_eval_pkgs/100k_memory6_base
TASK=cover_blocks
EVAL_NUM=8          # >= 2*num_envs, or the reset line carrying the counters never fires
OUT="${E2_OUT:-/job_data/eval_out}"
RD="${E2_RD:-/job_data/robodojo}"
LOG="$RUN_DIR/logs/stages.txt"
CAP="$RUN_DIR/logs/pod_capability.txt"
DRY="${E2_DRYRUN:-0}"

mkdir -p "$RUN_DIR/logs" "$OUT"
say() { echo "$*" | tee -a "$LOG"; }

{
  echo "=== pod $(date -u +%FT%TZ) uid=$(id -u) host=$(hostname) ==="
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader \
    2>/dev/null || echo "(no nvidia-smi)"
} | tee -a "$CAP"

if [ "$DRY" = "1" ]; then
  say "### DRY RUN -- skipping package check, tree patch and eval"
else
  # Assert the package really is memory-free. `MemoryVLAMemory` in the config
  # is the switch; without this check a copy-paste of the mem path would give
  # a clean-looking run that answers the opposite question.
  /usr/bin/python3 - "$BASE_PKG" <<'PY' | tee -a "$CAP" || { echo "FATAL baseline package check failed" | tee -a "$CAP"; exit 90; }
import json, sys
cfg = json.load(open(sys.argv[1] + "/model.config.json"))
has = "MemoryVLAMemory" in json.dumps(cfg)
print(f"OPEN OK pkg  MemoryVLAMemory={has} (expect False)")
raise SystemExit(1 if has else 0)
PY

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
  say "### stage $name start $(date -u +%FT%TZ) num_envs=$n pkg=baseline"
  rm -rf "$OUT"; mkdir -p "$OUT"

  local vram="$RUN_DIR/logs/vram_$name.csv"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 5 \
    > "$vram" 2>/dev/null &
  local sampler=$!
  [ -s "$vram" ] || echo 0 > "$vram"

  if [ "$DRY" = "1" ]; then
    mkdir -p "$OUT/w0"
    /usr/bin/python3 - "$OUT/w0" "$n" <<'FAKE'
import json, sys
d, n = sys.argv[1], int(sys.argv[2])
# A baseline package emits no bank_keys at all -- that absence is exactly what
# the parser must tolerate here and must not tolerate in E2c.
line = {"eval_episode": 2, "eval_forwards": 25 * n, "env_step": 800}
open(d + "/eval.log", "w").write(f"INFO policy reset: {line}\n")
break_it = __import__("os").environ.get("E2_DRYRUN_BREAK") == "1"
# Only the multi-env stage breaks: b1 has to stay usable as the reference,
# or the control compares against nothing.
score = 0.0 if (break_it and n > 1) else 0.05
json.dump({"details": {str(i): {"layout_id": i, "score": score}
                       for i in range(8)}}, open(d + "/_result.json", "w"))
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

forwards, banks = [], []
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
        for names in (d.get("bank_keys") or {}).values():
            if names:
                banks.append(sorted(names))

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

print(f"[prov {name}] num_envs={n} episodes_seen={len(forwards)} "
      f"peak_vram={peak}MiB banks_seen={banks or None}")
print(f"[prov {name}] scores={json.dumps(scores, sort_keys=True)}")

ok = True
if not forwards:
    print(f"[prov {name}] FAIL no episode reported")
    ok = False
# The inverse of E2c's check. A baseline package that grew a memory bank means
# the wrong package is loaded and the control answers nothing.
if banks:
    print(f"[prov {name}] FAIL a baseline package created memory banks "
          f"({banks}) -- this is not the control it claims to be")
    ok = False

ref = pathlib.Path(run_dir) / "logs" / "scores_ref.json"
if n == 1:
    ref.write_text(json.dumps(scores, sort_keys=True))
    print(f"[prov {name}] wrote reference scores")
elif ref.exists():
    r = json.loads(ref.read_text())
    live = [k for k, v in r.items() if v]
    dead = [k for k in live if not scores.get(k)]
    if not live:
        print(f"[prov {name}] VERDICT: INCONCLUSIVE -- the baseline scored 0 "
              "on every layout at num_envs=1 as well, so this control "
              "distinguishes nothing. Use a task the baseline can partially "
              "do, or compare against the mem package at num_envs=1.")
        print(f"[prov {name}] {'PASS' if ok else 'FAIL'}")
        sys.exit(0 if ok else 3)
    print(f"[prov {name}] VERDICT: of {len(live)} layouts scoring > 0 at "
          f"num_envs=1, {len(dead)} score 0 here -> "
          + ("(B) a memory-free policy collapses the same way, so the memory "
             "is NOT the cause -- look at RoboDojo's batch action path"
             if live and len(dead) == len(live) else
             "(A) the baseline survives multi-env, so the collapse is "
             "specific to the memory path"))
    print(f"[prov {name}] per-layout: "
          + json.dumps({k: [r.get(k), scores.get(k)] for k in sorted(r)}))
print(f"[prov {name}] {'PASS' if ok else 'FAIL'}")
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

run_stage b1 1
run_stage b2 2

say "ALL STAGES DONE rc_sum=$RC_ALL $(date -u +%FT%TZ)"
exit "$RC_ALL"
