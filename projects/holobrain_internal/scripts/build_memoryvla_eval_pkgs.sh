#!/usr/bin/env bash
# Pull the four trained checkpoints off PFS and assemble the flat deploy
# packages evaluation loads.
#
# Not assemble_deploy_package.sh: that one takes an export.py workspace and
# hard-asserts MemoryVLAMemory, which is exactly wrong for the two baseline
# packages. The job's own output/ already carries the processor, the inference
# config and the urdf, so the flat layout is a copy job, not an export.
#
# Landing: packages are MODEL-side artifacts, so they go under the model's
# project root (robo_orchard_lab), not under the benchmark's.
#
# The stride32 row points at a run that is still training as of 2026-08-13; its
# checkpoint_18 appears only when the job reaches 90k steps. Until then that row
# fails on a missing content-length, which is the intended behaviour -- fetch()
# refuses a short read rather than assembling half a package. Use
# ROWS_FILTER=<name> to build just one.
set -eo pipefail

OUT=/jfs-public/users/kun01.wu/robo_orchard_lab/port/memoryvla/eval_pkgs
VLM=/horizon-bucket/robot_lab/users/xuewu.lin/ckpt
DS=robodojo_arx_x5a
B_BCLOUD=http://pfs-svcspawner.bcloud-bj-zone1.hobot.cc/user/homespace/kun01.wu-labs/plat_gpu
B_ACLOUD=http://cpfs-svcspawner.acloud.hobot.cc/user/homespace/kun01.wu-labs/plat_gpu

# name | <output root url> | checkpoint | expect_memoryvla
#
# Each row carries its own root rather than sharing one BASE: the host differs
# per cluster (pfs- on bcloud, cpfs- on acloud) and so does the date, so a run
# on another cluster could not be expressed at all before.
#
# The root is $B_<cluster>/<YYYY-MM-DD>/<HH-MM>/<job_id>/<job_name>/output --
# every part of which is in the job's log_url, and the job_id is also recoverable
# from a pod's hostname (<job_id>-task-N) when the submit tool fails to capture
# it. Run with ROWS_FILTER=<name> to build a single package.
# NOTE: no 100k_memory6_base_ck19 row. The base run's checkpoint_19 holds
# model.config.json and no model.safetensors (404), so its last usable
# checkpoint is 18 at 95,000 steps. A row here would fail on every build.
ROWS="
15k_conveyor_mem|$B_BCLOUD/2026-08-07/10-38/bcloud-bj-zone1-cb5a332fce15/holobrain_robodojo_mvla_15k_conveyor_mem_ef565d74_9208_11f1_b662_02f34d1460a1/output|6|yes
15k_conveyor_base|$B_BCLOUD/2026-08-07/10-40/bcloud-bj-zone1-875062c3b100/holobrain_robodojo_mvla_15k_conveyor_base_4722995a_9209_11f1_a630_02f34d1460a1/output|6|no
100k_memory6_mem|$B_BCLOUD/2026-08-07/10-42/bcloud-bj-zone1-321cccff92d6/holobrain_robodojo_mvla_100k_memory6_mem_9fd584ae_9209_11f1_b677_02f34d1460a1/output|18|yes
100k_memory6_mem_ck19|$B_BCLOUD/2026-08-07/10-42/bcloud-bj-zone1-321cccff92d6/holobrain_robodojo_mvla_100k_memory6_mem_9fd584ae_9209_11f1_b677_02f34d1460a1/output|19|yes
100k_memory6_base|$B_BCLOUD/2026-08-07/10-45/bcloud-bj-zone1-db163aa38027/holobrain_robodojo_mvla_100k_memory6_base_f8df9ab2_9209_11f1_bdfb_02f34d1460a1/output|18|no
100k_memory6_mem_stride32|$B_ACLOUD/2026-08-13/10-36/acloud-ad66079d5082/holobrain_robodojo_mvla_100k_memory6_mem_stride32_h20_a66c8906_96bf_11f1_961c_02f34d1460a1/output|18|yes
"

echo "=== 目标盘余量 ==="
df -h "$OUT" 2>/dev/null || df -h /jfs-public
mkdir -p "$OUT"

# fetch <url> <dest> -- refuses a short read; a truncated safetensors would
# otherwise surface as a confusing load error hours later.
fetch() {
  local url=$1 dest=$2
  local want
  want=$(curl -sSI --max-time 60 "$url" | tr -d '\r' \
         | awk 'tolower($1)=="content-length:"{print $2}' | tail -1)
  [ -n "$want" ] || { echo "  !! 拿不到 content-length: $url"; return 1; }
  if [ -f "$dest" ] && [ "$(stat -c %s "$dest")" = "$want" ]; then
    echo "  = $(basename "$dest") 已存在且字节数相符 ($want)"; return 0
  fi
  curl -sS --max-time 3600 -o "$dest.part" "$url"
  local got; got=$(stat -c %s "$dest.part")
  if [ "$got" != "$want" ]; then
    echo "  !! $(basename "$dest") 短读: got=$got want=$want"; rm -f "$dest.part"; return 1
  fi
  mv "$dest.part" "$dest"
  echo "  + $(basename "$dest") $got bytes"
}

echo "$ROWS" | grep -v '^$' | while IFS='|' read -r name ROOT ck expect; do
  if [ -n "${ROWS_FILTER:-}" ] && [ "$name" != "$ROWS_FILTER" ]; then continue; fi
  D="$OUT/$name"
  echo
  echo "########## $name  (checkpoint_$ck, memoryvla=$expect)"
  mkdir -p "$D/urdf"
  fetch "$ROOT/checkpoints/checkpoint_$ck/model.safetensors" "$D/model.safetensors"
  fetch "$ROOT/checkpoints/checkpoint_$ck/model.config.json" "$D/model.config.json"
  fetch "$ROOT/${DS}_processor.json"                          "$D/${DS}_processor.json"
  fetch "$ROOT/${DS}_inference.config.json"                   "$D/${DS}_inference.config.json"
  fetch "$ROOT/urdf/robotwin2_dual_arm_arx_x5a.urdf"          "$D/urdf/robotwin2_dual_arm_arx_x5a.urdf"
  ln -sfn "$VLM" "$D/ckpt"

  python3 - "$D" "$DS" "$expect" <<'PY'
import json, sys, os
d, ds, expect = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = json.dumps(json.load(open(f"{d}/model.config.json")))
has = "MemoryVLAMemory" in cfg
want = expect == "yes"
if has != want:
    raise SystemExit(f"  !! FAIL MemoryVLAMemory={has} 但期望 {want}")
proc = json.dumps(json.load(open(f"{d}/{ds}_processor.json")))
if want and '"step_index"' not in proc:
    raise SystemExit("  !! FAIL processor 缺 step_index")
sz = os.path.getsize(f"{d}/model.safetensors")
if sz < 1_000_000_000:
    raise SystemExit(f"  !! FAIL 权重只有 {sz} 字节")
print(f"  OK  MemoryVLAMemory={has}(期望{want})  权重 {sz/1e9:.2f} GB  ckpt->{os.path.realpath(d+'/ckpt')}")
PY
done

echo
echo "=== 汇总 ==="
du -sh "$OUT"/* 2>/dev/null
