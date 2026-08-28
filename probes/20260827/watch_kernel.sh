#!/bin/bash
# 等 kernel 跑完 → 核验产物 → 打印线上25局成绩。不自动提交, 提交要用户拍板。
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
K=~/.local/bin/kaggle; REF=jinbowang1/arc3-duck-q38-lb9
OUT=~/Desktop/project/arc-agi-3/probes/20260827/lb9_v2_out; mkdir -p $OUT
while true; do
  st=$($K kernels status $REF 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z]*')
  echo "[$(date '+%H:%M:%S')] $st"
  case "$st" in *COMPLETE*) break;; *ERROR*|*CANCEL*) echo "❌ kernel 失败"; exit 1;; esac
  sleep 300
done
files=$($K kernels files $REF --page-size 200 2>&1)
echo "$files" | grep -q 'submission.parquet' && echo "✅ submission.parquet 在" || { echo "❌ 没有 parquet"; exit 1; }
cd $OUT && $K kernels output $REF -p . --file-pattern '\.log$' --page-size 200 >/dev/null 2>&1
L=$(ls *.log 2>/dev/null | head -1)
[ -n "$L" ] && { echo "── 线上 25 局成绩 ──"; grep -E 'mean score|median score|total actions' "$L" | tail -3; }
echo "⏸ 已就绪, 等用户拍板才提交"
