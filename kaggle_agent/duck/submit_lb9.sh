#!/bin/bash
# 等 arc3-duck-q38-lb9 跑完 → 核图/核parquet/看成绩 → 提交(用户 08-26 18:40 已拍板"跑完就提交")
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
K=~/.local/bin/kaggle; REF=jinbowang1/arc3-duck-q38-lb9; OUT=~/Desktop/project/arc-agi-3/probes/20260826/lb9_out; mkdir -p $OUT
log(){ echo "[$(date '+%H:%M:%S')] $*"; }
while true; do st=$($K kernels status $REF 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z]*'); log "status=$st"
  case "$st" in *COMPLETE*) break;; *ERROR*|*CANCEL*) log "❌ kernel 失败: $st"; exit 1;; esac; sleep 300; done
files=$($K kernels files $REF --page-size 200 2>&1); echo "$files" | grep -q 'submission.parquet' && log "✅ submission.parquet 在" || { log "❌ 没有 submission.parquet"; echo "$files" | head; exit 1; }
cd $OUT; $K kernels output $REF -p . --file-pattern 'transcripts/sb26' --page-size 200 >/dev/null 2>&1; $K kernels output $REF -p . --file-pattern '\.log$' --page-size 200 >/dev/null 2>&1
T=$(ls transcripts/*.txt 2>/dev/null | head -1); L=$(ls *.log 2>/dev/null | head -1)
if [ -n "$T" ]; then n=$(awk '/\[SYSTEM PROMPT\]/{p=1} /\[USER PROMPT\]/{exit} p' "$T" | grep -c 'attached image'); [ "$n" -ge 1 ] && log "✅ 带图 (system prompt 有 attached image)" || log "⚠️ transcript 里没看到 attached image"; grep -m1 -o 'model: [^ ]*' "$T"; else log "⚠️ 没拉到 transcript"; fi
[ -n "$L" ] && { log "成绩: $(grep 'mean score' "$L" | tail -1) $(grep 'total actions' "$L" | tail -1)"; grep -qi 'Traceback\|CUDA error\|OutOfMemory' "$L" && log "⚠️ 日志有 Traceback/CUDA 错误"; }
$K competitions submit arc-prize-2026-arc-agi-3 -k $REF -v 1 -f submission.parquet -m "LB-9 replica: duck harness + Qwen3.8-27B FP8 + concurrency 28 (multimodal)" 2>&1 | tail -2
sleep 20; $K competitions submissions arc-prize-2026-arc-agi-3 2>&1 | head -4; log DONE
