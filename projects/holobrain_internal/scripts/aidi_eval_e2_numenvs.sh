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
EVAL_NUM=4
OUT=/job_data/eval_out
RD=/job_data/robodojo
LOG="$RUN_DIR/logs/stages.txt"
CAP="$RUN_DIR/logs/pod_capability.txt"

mkdir -p "$RUN_DIR/logs" "$OUT"
say() { echo "$*" | tee -a "$LOG"; }

{
  echo "=== pod $(date -u +%FT%TZ) uid=$(id -u) host=$(hostname) ==="
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
} | tee -a "$CAP"

/usr/bin/python3 -c "
open('$BASE_PKG/model.config.json', 'rb').read(64); print('OPEN OK pkg')
" | tee -a "$CAP" || { echo "FATAL pkg unreadable" | tee -a "$CAP"; exit 90; }

# The image's RoboDojo caps num_envs at 1 and never forwards --num_envs, so it
# has to be patched. Exits 91 if an anchor moved -- running unpatched here would
# silently give num_envs=1 for every stage and look like a clean negative.
say "### patching RoboDojo tree $(date -u +%FT%TZ)"
bash "${WORK}/robodojo_pod_tree.sh" "$RD" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
[ "$rc" = "0" ] || { say "FATAL robodojo_pod_tree.sh rc=$rc -- refusing to run unpatched"; exit "$rc"; }
grep -q ROBODOJO_NUM_ENVS "$RD/src/eval_client/main.py" \
  || { say "FATAL patch assertion failed after a clean rc"; exit 91; }

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
  RC_ALL=$((RC_ALL + rc))

  kill "$sampler" 2>/dev/null || true

  /usr/bin/python3 - "$OUT" "$name" "$n" "$vram" "$RUN_DIR" <<'PY' | tee -a "$LOG"
import ast, json, pathlib, sys
out, name, n, vram, run_dir = sys.argv[1:6]
n = int(n)

keys, reads, forwards = set(), [], []
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
        # bank_lengths is {bank_name: [len, ...]} -- one entry per live key, so
        # its length is how many episodes the bank is holding at once. That is
        # the only place separate-banks shows up as an observable.
        for lens in (d.get("bank_lengths") or {}).values():
            if lens:
                keys.add(len(lens))

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

print(f"[prov {name}] num_envs={n} concurrent_bank_keys={sorted(keys) or None} "
      f"peak_vram={peak}MiB episodes_seen={len(forwards)} "
      f"min_history_reads={min(reads) if reads else None}")
print(f"[prov {name}] scores={json.dumps(scores, sort_keys=True)}")

ref = pathlib.Path(run_dir) / "logs" / "scores_ref.json"
ok = True
if not forwards:
    print(f"[prov {name}] FAIL no episode reported")
    ok = False
if n > 1:
    if not keys or max(keys) < n:
        print(f"[prov {name}] FAIL banks never held {n} episodes at once "
              f"(saw {sorted(keys) or None}) -- the envs shared one bank, or "
              "num_envs never actually reached the sim")
        ok = False
    if reads and min(reads) == 0:
        print(f"[prov {name}] FAIL an env retrieved history 0 times, so its "
              "memory contributed nothing")
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
PY

  cp -r "$OUT/." "$RUN_DIR/" 2>&1 | tail -3 || true
  say "### stage $name done rc=$rc $(date -u +%FT%TZ)"
}

run_stage n1 1
run_stage n2 2
run_stage n4 4

say "ALL STAGES DONE rc_sum=$RC_ALL $(date -u +%FT%TZ)"
exit "$RC_ALL"
