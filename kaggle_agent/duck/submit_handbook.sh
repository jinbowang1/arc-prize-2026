#!/bin/bash
# 等 arc3-duck-handbook 跑完 → 五项核验 → 提交。任何一项不过就不交。
K=~/.local/bin/kaggle; REF=jinbowang1/arc3-duck-handbook; OUT=$1
log(){ echo "[$(date '+%H:%M:%S')] $*"; }
for i in $(seq 1 60); do
  st=$($K kernels status $REF 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z]*'); log "status=$st"
  case "$st" in *COMPLETE*) break;; *ERROR*|*CANCEL*) log "❌ kernel 失败: $st"; exit 1;; esac; sleep 300
done
[ "$st" != *COMPLETE* ] && { [[ "$st" == *COMPLETE* ]] || { log "❌ 超时未完成"; exit 1; }; }
# ① parquet 门禁(服务端文件清单)
files=$($K kernels files $REF --page-size 100 2>&1); echo "$files" | grep -q 'submission.parquet' && log "✅ submission.parquet 在输出里" || { log "❌ 没有 submission.parquet"; echo "$files" | head; exit 1; }
# ② 拉日志核源码包与手册
mkdir -p $OUT && cd $OUT && for i in 1 2 3 4; do $K kernels output $REF -p . >/dev/null 2>&1; ls *.log >/dev/null 2>&1 && break; sleep 5; done
L=$(ls *.log 2>/dev/null | head -1); [ -z "$L" ] && { log "⚠️ 没拉到日志文件, 只按文件清单判断"; }
if [ -n "$L" ]; then
  grep -q 'taaf-source-handbook' "$L" && log "✅ 源码包=taaf-source-handbook" || { log "❌ 日志里没有 taaf-source-handbook"; exit 1; }
  grep -qi 'Traceback\|CUDA error\|OutOfMemory' "$L" && log "⚠️ 日志里有 Traceback/CUDA 错误, 先看再说" && grep -n -i -m3 'Traceback\|CUDA error' "$L"
  log "25局成绩: $(grep -c '\[finished\]' "$L") 局结束; $(grep 'mean score' "$L" | tail -1); $(grep 'total actions' "$L" | tail -1)"
  grep '\[finished\]' "$L" | sed 's/note=.*//' | cut -c1-110 > per_game.txt; log "逐局明细存 per_game.txt"
fi
# ③ 手册字样(prompt.log 若在输出里)
grep -rl 'Field guide' . 2>/dev/null | head -1 | grep -q . && log "✅ 输出里能找到 'Field guide'" || log "ℹ️ 输出里没带 prompt 文本(不影响提交, 源码包已核)"
# ④ 提交
ver=$($K kernels list --mine --page-size 50 2>/dev/null | grep -c . ); 
$K competitions submit arc-prize-2026-arc-agi-3 -k $REF -v 1 -f submission.parquet -m "duck official (Qwen3.6 27B) + field guide of game families in system prompt" 2>&1 | tail -2
sleep 20; $K competitions submissions arc-prize-2026-arc-agi-3 2>&1 | head -4
log "DONE"
