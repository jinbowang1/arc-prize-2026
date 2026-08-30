#!/bin/bash
# 等 yield=300 的 kernel 跑完 → 核验参数真的生效 → 打印成绩。不自动提交。
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
K=~/.local/bin/kaggle; REF=jinbowang1/arc3-duck-q38-lb9
OUT=~/Desktop/project/arc-agi-3/probes/20260827/y300_out; mkdir -p $OUT
while true; do
  st=$($K kernels status $REF 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z]*')
  echo "[$(date '+%H:%M:%S')] $st"
  case "$st" in *COMPLETE*) break;; *ERROR*|*CANCEL*) echo "❌ kernel 失败"; exit 1;; esac
  sleep 300
done
$K kernels files $REF --page-size 200 2>&1 | grep -q 'submission.parquet' && echo "✅ parquet 在" || { echo "❌ 无 parquet"; exit 1; }
cd $OUT
$K kernels output $REF -p . --file-pattern '\.log$' --page-size 200 >/dev/null 2>&1
for g in sb26 ft09; do $K kernels output $REF -p . --file-pattern "transcripts/$g" --page-size 200 >/dev/null 2>&1; done
L=$(ls *.log 2>/dev/null | head -1)
echo "── 参数核验 ──"
grep -o "patched LOCAL_ANALYZER_YIELD_SECONDS -> [0-9]*" "$L" 2>/dev/null | head -1 || echo "⚠️ 日志里没看到 patch 打印"
T=$(ls transcripts/*.txt 2>/dev/null | head -1)
[ -n "$T" ] && grep -m1 -o "yield_seconds: [0-9.]*" "$T" || echo "⚠️ 没拉到 transcript"
echo "── 成绩 ──"
[ -n "$L" ] && grep -E 'mean score|median score|total actions' "$L" | tail -3
echo "⏸ 已就绪, 等拍板才提交"
