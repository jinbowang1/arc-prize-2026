#!/bin/bash
# 等 arc3-v31-y300 跑完 → 拉输出 → 核验(YIELD=300 / MTP 参数生效 / 有无回退) → 算机制指标。不自动提交。
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
K=~/.local/bin/kaggle; REF=jinbowang1/arc3-v31-replica; OUT=~/Desktop/project/arc-agi-3/probes/20260830/v31rep_out; mkdir -p $OUT; cd $OUT
while true; do st=$($K kernels status $REF 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z]*'); echo "[$(date '+%H:%M:%S')] $st"
  case "$st" in *COMPLETE*) break;; *ERROR*|*CANCEL*) echo "❌ kernel 失败"; exit 1;; esac; sleep 300; done
$K kernels files $REF --page-size 300 2>&1 | grep -q 'submission.parquet' && echo "✅ 有 submission.parquet" || echo "❌ 无 parquet"
$K kernels output $REF -p . --file-pattern '\.log$' --page-size 300 >/dev/null 2>&1
for g in ar25 bp35 cd82 cn04 dc22 ft09 g50t ka59 lf52 lp85 ls20 m0r0 r11l re86 s5i5 sb26 sc25 sk48 sp80 su15 tn36 tr87 tu93 vc33 wa30; do
  for try in 1 2 3; do f=$(ls transcripts/${g}*_p0.txt 2>/dev/null | head -1); [ -n "$f" ] && [ -s "$f" ] && break
    $K kernels output $REF -p . --file-pattern "transcripts/${g}" --page-size 300 >/dev/null 2>&1; done; done
echo "════ 核验 ════"
T=$(find transcripts -name '*.txt' -size +1k 2>/dev/null | head -1); echo "  yield_seconds = $(grep -m1 -o 'yield_seconds: [0-9.]*' "$T")"
L=$(ls *.log 2>/dev/null | grep -v vllm | head -1); grep -o 'patched LOCAL_ANALYZER_YIELD_SECONDS[^"\\]*\|V31: launching[^"\\]*\|falling back[^"\\]*\|V31 watchdog[^"\\]*' "$L" | sort | uniq -c
grep -o 'speculative[^"\\]\{0,80\}\|async-scheduling\|no-enable-chunked-prefill' vllm-openai-server.log 2>/dev/null | sort | uniq -c | head
echo "════ 成绩(仅参考) ════"; grep -o 'mean score[^"\\]*\|total actions[^"\\]*' "$L" | tail -2
echo "════ 机制指标 ════"; cd ~/Desktop/project/arc-agi-3 && sed "s#probes/20260826/lb9_out#probes/20260830/v31rep_out#g" probes/20260827/diag.py > /tmp/diag_v31rep.py && python3 /tmp/diag_v31rep.py 2>&1 | sed -n '/=== 合计/,$p'
echo "⏸ 已停在提交前, 等拍板"
