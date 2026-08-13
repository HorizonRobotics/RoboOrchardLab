#!/usr/bin/env bash
# Make a writable, patched RoboDojo tree for a pod to run from.
#
# Why this exists: robodojo_eval.py defaults --robodojo_root to /opt/robodojo,
# which comes from the docker image, and the eval submit configs upload no
# RoboDojo path. So every edit to ~/git_repo/RoboDojo has applied to dev-box
# runs only. That includes the PhysX watchdog fix -- verified against the image,
# whose main.py still gates the monitor on Articulation, the exact form that
# cost a 7.5-hour slot.
#
# /opt/robodojo is root:root drwxr-xr-x and a pod runs as an ordinary user, so
# it cannot be patched in place; it is 142 MB (140 of which is third_party), so
# copying it is seconds.
#
#   bash robodojo_pod_tree.sh [DEST]      # DEST default /job_data/robodojo
#   ROBODOJO_SRC=... to copy from somewhere other than /opt/robodojo
#
# then pass --robodojo_root DEST to robodojo_eval.py.
#
# Every patch asserts its anchor before and its result after. A future image
# bump that moves one of these lines must fail loudly here rather than run
# unpatched -- silently running the old code is the failure this whole script
# is about.
set -uo pipefail

SRC="${ROBODOJO_SRC:-/opt/robodojo}"
DEST="${1:-/job_data/robodojo}"

[ -d "$SRC" ] || { echo "FATAL no RoboDojo tree at $SRC"; exit 90; }
mkdir -p "$DEST" || { echo "FATAL cannot create $DEST"; exit 90; }

echo "[tree] copying $SRC -> $DEST ($(du -sh "$SRC" 2>/dev/null | cut -f1))"
cp -a "$SRC/." "$DEST/" || { echo "FATAL copy failed"; exit 90; }

python3 - "$DEST" <<'PY' || exit 91
import pathlib
import sys

dest = pathlib.Path(sys.argv[1])

# (anchor, replacement, a string that must be present afterwards)
PATCHES = [
    (
        "src/eval_client/main.py",
        'parser.add_argument("--num_envs", type=int, default=1, '
        'help="Number of environments to spawn.")',
        'parser.add_argument(\n'
        '    "--num_envs",\n'
        '    type=int,\n'
        '    default=int(os.environ.get("ROBODOJO_NUM_ENVS", 1)),\n'
        '    help="Number of environments to spawn. Default from '
        '$ROBODOJO_NUM_ENVS: the\\n"\n'
        '    "launch chain passes a fixed argument list and never forwarded '
        'this one.",\n'
        ')',
        'ROBODOJO_NUM_ENVS',
    ),
    (
        # Behaviour only. The now-unused _physx_monitor_needed stays; patching
        # a call site is a far more stable anchor than a function body.
        "src/eval_client/main.py",
        "enable_monitor = _physx_monitor_needed(args_cli.task_name)",
        'enable_monitor = (\n'
        '    os.environ.get("ROBODOJO_PHYSX_MONITOR", "1").strip()\n'
        '    not in ("0", "false", "False")\n'
        ')  # was gated on Articulation; conveyor declares none, so the\n'
        '   # watchdog was off when PhysX corrupted its scene and the process\n'
        '   # sat at 126% CPU for 7h33m producing nothing.',
        'ROBODOJO_PHYSX_MONITOR',
    ),
    (
        "XPolicyLab/utils/setup_env_client.sh",
        'protocol="${protocol_override:-${yaml_protocol}}"',
        'protocol="${protocol_override:-${yaml_protocol}}"\n'
        '\n'
        '# main.py caps num_envs at 1 while eval_batch is false, so a policy\n'
        '# with per-env memory needs this too. Unset leaves the yaml value.\n'
        'eval_batch="${ROBODOJO_EVAL_BATCH:-${eval_batch}}"',
        'ROBODOJO_EVAL_BATCH',
    ),
]

fails = []
for rel, anchor, repl, expect in PATCHES:
    path = dest / rel
    if not path.exists():
        fails.append(f"{rel}: missing from the tree")
        continue
    text = path.read_text()
    if expect in text:
        print(f"[patch] {rel}: already carries {expect}, skipping")
        continue
    if anchor not in text:
        fails.append(
            f"{rel}: anchor absent -- {anchor[:70]!r}. The image moved this "
            "line; fix the anchor rather than running unpatched."
        )
        continue
    path.write_text(text.replace(anchor, repl, 1))
    if expect not in path.read_text():
        fails.append(f"{rel}: wrote the patch but {expect} is not there")
    else:
        print(f"[patch] {rel}: applied, {expect} present")

if fails:
    print("PATCH FAILED:")
    for f in fails:
        print("  -", f)
    raise SystemExit(1)
print("[patch] all applied")
PY

# The tree has to survive being imported, not just edited.
python3 -c "
import ast, sys
ast.parse(open('$DEST/src/eval_client/main.py').read())
print('[tree] main.py parses')
" || { echo "FATAL patched main.py does not parse"; exit 91; }
bash -n "$DEST/XPolicyLab/utils/setup_env_client.sh" \
  || { echo "FATAL patched setup_env_client.sh does not parse"; exit 91; }
echo "[tree] setup_env_client.sh parses"
echo "[tree] ready: $DEST"
