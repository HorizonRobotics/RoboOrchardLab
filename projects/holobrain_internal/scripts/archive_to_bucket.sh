#!/bin/bash
# 把一份 job 产出归档到 bucket 的项目根，落点和命名由本脚本强制，避免每次手写 cp
# 又散落到 aidi_output/ 之类的地方去。
#
#   archive_to_bucket.sh ckpts        <run_name> <src_dir>
#   archive_to_bucket.sh eval_results <run_name> <src_dir>
#
# 例：
#   archive_to_bucket.sh ckpts holobrain_robodojo_posttrain_v10/checkpoint_50000 ~/tmp/ckpt50k
#
# 为什么需要它：AIDI **不会**把 /job_data 自动同步到 bucket（`users/<user>-labs/`
# 那个路径不存在），所以每个 job 跑完都得自己归档。散手 cp 的结果就是产物落得到处都是。
#
# 做的事：校验参数 → 拒绝越出项目根 → cp -rd（保留 symlink，绝不跟随，
# 否则会把 ckpt -> xuewu.lin/ckpt 那一大坨拷进来）→ 逐文件 md5 双边校验 → 写清单。

set -e

KIND=$1
NAME=$2
SRC=$3

BUCKET_ROOT=/horizon-bucket/robot_lab/users/kun01.wu/robo_orchard_lab
JFS_ROOT=/jfs-public/users/kun01.wu/robo_orchard_lab
TV=$JFS_ROOT/tools/tree_verify.py

usage() {
  echo "用法: $(basename "$0") {ckpts|eval_results} <run_name> <src_dir>"
  echo
  echo "  ckpts        定版 deploy package，评测 job 的 --model_dir 会读它"
  echo "  eval_results 评测结果归档"
  echo
  echo "落点: $BUCKET_ROOT/<kind>/<run_name>"
  exit 2
}

# 用 if 而不是 `[ ] || [ ] && usage`：后者在三个参数都给全时整条返回非零，
# 配合 set -e 会让脚本在这里静默退出。
if [ -z "$KIND" ] || [ -z "$NAME" ] || [ -z "$SRC" ]; then
  usage
fi
case "$KIND" in
  ckpts|eval_results) ;;
  *) echo "!! kind 只能是 ckpts 或 eval_results，收到: $KIND"; usage ;;
esac
[ -d "$SRC" ] || { echo "!! 源目录不存在: $SRC"; exit 1; }
[ -f "$TV" ] || { echo "!! 缺校验工具 $TV"; exit 1; }

# 拒绝 ../ 之类把文件写到项目根外面去
case "$NAME" in
  /*|*..*) echo "!! run_name 不能是绝对路径、也不能含 ..: $NAME"; exit 1 ;;
esac

DST=$BUCKET_ROOT/$KIND/$NAME
echo "SRC  $SRC"
echo "DST  $DST"
SAMPLE=$(find "$SRC" -type f -printf '%P\n' 2>/dev/null | head -1)
echo "样本文件最终路径  $DST/$SAMPLE"
echo "源里的 symlink（会原样保留，不跟随）:"
find "$SRC" -type l -printf '  %P -> %l\n' 2>/dev/null | head
du -sh "$SRC"

if [ -e "$DST" ]; then
  echo
  echo "目标已存在，先比对是否一致 ..."
  if python3 "$TV" "$SRC" "$DST" --jobs 6 --mode equal >/dev/null 2>&1; then
    echo "已经归档过且内容一致，无需重做。"
    exit 0
  fi
  echo "!! 目标已存在但内容不同。bucket 上不要就地覆盖历史产物 ——"
  echo "!! 换一个 run_name，或先确认旧的可以丢再用 aidi-inf-cli bkt-file rm 删掉。"
  exit 1
fi

echo
read -r -p "确认归档？输入 yes 继续: " a
[ "$a" = "yes" ] || { echo "已取消。"; exit 0; }

mkdir -p "$(dirname "$DST")"
t0=$(date +%s)
# -d 保留 symlink；-T 消除 "DST 不存在时会多套一层同名目录" 的歧义
cp -rdT --preserve=mode,timestamps "$SRC" "$DST"
t1=$(date +%s)
BYTES=$(du -sb "$DST" | cut -f1)
echo "拷完 ${BYTES}B / $((t1 - t0))s"

mkdir -p "$JFS_ROOT/manifests"
MAN=$JFS_ROOT/manifests/archive_$(echo "$KIND/$NAME" | tr / _).md5
python3 "$TV" "$SRC" "$DST" --jobs 6 --mode equal --manifest "$MAN"

echo
echo "归档完成: $DST"
echo "md5 清单:  $MAN"
echo
echo "如果这是要拿去评测的 ckpt，把 submit cfg 里的 --model_dir 指到:"
echo "  $DST/"
