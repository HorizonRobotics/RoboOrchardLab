#!/usr/bin/env bash
# Turn an export.py workspace into the flat package evaluation actually loads.
#
# export.py writes processors at the top level and the model under model/,
# with a ckpt/ directory whose inner symlink points at THIS machine's
# checkout. robodojo_eval.py wants the layout the v9 packages have: one flat
# directory, with ckpt a single symlink to a path a pod can see. Copying the
# export tree as-is gives a package that resolves on the dev box and dangles
# in the pod.
#
#   assemble_deploy_package.sh <export-workspace> <out-dir> [dataset-name]
#
# The VLM base is read-only and belongs to another user; VLM_CKPT only ever
# becomes the target of a symlink, nothing is written under it. Pass it in:
#
#   VLM_CKPT=<vlm base dir> assemble_deploy_package.sh ...
#
# Never removes anything: refuses if <out-dir> is non-empty, so it can be
# pointed at a bucket path without needing the deletes it would not get.
set -euo pipefail

SRC=${1:?export workspace}
OUT=${2:?output package dir}
DS=${3:-robodojo_arx_x5a}
: "${VLM_CKPT:?set VLM_CKPT to the VLM base directory}"

if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
  echo "REFUSING: $OUT exists and is not empty" >&2
  exit 1
fi
mkdir -p "$OUT/urdf"

install -m 644 "$SRC/model/model.safetensors"           "$OUT/"
install -m 644 "$SRC/model/model.config.json"           "$OUT/"
install -m 644 "$SRC/${DS}_processor.json"              "$OUT/"
install -m 644 "$SRC/model/${DS}_inference.config.json" "$OUT/"
install -m 644 "$SRC"/model/urdf/*                      "$OUT/urdf/"
ln -s "$VLM_CKPT" "$OUT/ckpt"

echo "--- assembled ---"
ls -la "$OUT"

# The two things that make it a *memoryvla* package rather than a baseline
# one. Cheap, and the alternative is finding out from a success rate.
python3 - "$OUT" "$DS" <<'PY'
import json
import sys

out, ds = sys.argv[1], sys.argv[2]
cfg = json.dumps(json.load(open("%s/model.config.json" % out)))
if "MemoryVLAMemory" not in cfg:
    raise SystemExit("FAIL: model.config.json has no MemoryVLAMemory")
proc = json.dumps(json.load(open("%s/%s_processor.json" % (out, ds))))
if '"step_index"' not in proc:
    raise SystemExit("FAIL: %s_processor.json has no step_index" % ds)
print("OK: MemoryVLAMemory in model.config.json, step_index in processor")
PY
