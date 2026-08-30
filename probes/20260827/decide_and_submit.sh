#!/bin/bash
# kernel 跑完后: 核验参数 → 算机制指标 → 满足条件才提交。
# 判据(08-28 定): 公开集 mean 无预测力(2.74→4.69 而隐藏集 1.92→1.85), 故只作参考。
#   必要条件: transcript 里 yield_seconds 确实是 300
#   主判据:   零动作轮下降 且 每轮动作数上升(与第二层 -11.2pp / +0.12 同方向)
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
K=~/.local/bin/kaggle; REF=jinbowang1/arc3-duck-q38-lb9
OUT=~/Desktop/project/arc-agi-3/probes/20260827/y300_out; mkdir -p $OUT; cd $OUT

while true; do
  st=$($K kernels status $REF 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z]*')
  echo "[$(date '+%H:%M:%S')] $st"
  case "$st" in *COMPLETE*) break;; *ERROR*|*CANCEL*) echo "❌ kernel 失败, 不提交"; exit 1;; esac
  sleep 300
done

$K kernels files $REF --page-size 200 2>&1 | grep -q 'submission.parquet' || { echo "❌ 无 parquet"; exit 1; }
$K kernels output $REF -p . --file-pattern '\.log$' --page-size 200 >/dev/null 2>&1
for g in $(cat /tmp/gids.txt 2>/dev/null || echo "sb26 ft09 vc33 r11l cd82 ls20"); do
  for try in 1 2 3; do
    f="transcripts/${g}_p0.txt"; [ -s "$f" ] && break
    $K kernels output $REF -p . --file-pattern "transcripts/${g}" --page-size 200 >/dev/null 2>&1
  done
done

echo "════ 参数核验 ════"
T=$(find transcripts -name '*.txt' -size +1k 2>/dev/null | head -1)
if [ -z "$T" ]; then echo "❌ 没拉到 transcript, 无法核验 → 不自动提交"; exit 1; fi
YS=$(grep -m1 -o "yield_seconds: [0-9.]*" "$T" | grep -o '[0-9.]*')
echo "  transcript 里 yield_seconds = ${YS:-未找到}"
case "$YS" in 300|300.0) echo "  ✅ 参数生效";; *) echo "  ❌ 参数未生效(应为300) → 不提交"; exit 1;; esac

echo "════ 机制指标 ════"
python3 - <<'PY'
import re,glob,os
hp=re.compile(r'^--- analysis_step=(\d+) \| action=(\d+) \|')
turns=acts=zero=0
for f in glob.glob('transcripts/*.txt'):
    if os.path.getsize(f)<1024: continue
    t=open(f,encoding='utf-8',errors='replace').read()
    segs=re.split(r'(?=^--- analysis_step=)',t,flags=re.M)[1:]
    hs=[hp.match(s) for s in segs]
    if not hs or not hs[0]: continue
    d=[int(hs[i+1].group(2))-int(hs[i].group(2)) for i in range(len(hs)-1) if hs[i] and hs[i+1]]
    turns+=len(segs); acts+=max([int(h.group(2)) for h in hs if h]+[0]); zero+=sum(1 for x in d if x==0)
dn=max(1,turns-1)
zr=100*zero/dn; apt=acts/max(1,turns)
print(f"  零动作轮 {zr:.1f}%   (第六发基线 57.8%)")
print(f"  每轮动作数 {apt:.2f}   (第六发基线 1.45)")
ok = zr < 57.8 and apt > 1.45
print(f"  判定: {'✅ 机制指标同方向, 可提交' if ok else '✗ 机制指标未改善'}")
open('/tmp/y300_verdict','w').write('GO' if ok else 'NOGO')
PY

L=$(ls *.log 2>/dev/null|head -1)
echo "════ 公开集成绩(仅供参考, 无预测力) ════"
[ -n "$L" ] && grep -E 'mean score|median score|total actions' "$L" | tail -3

if [ "$(cat /tmp/y300_verdict 2>/dev/null)" = "GO" ]; then
  echo "════ 提交 ════"
  $K competitions submit arc-prize-2026-arc-agi-3 -k $REF -v 3 -f submission.parquet \
    -m "yield_seconds 60->300: let act-look-act actually happen on Kaggle single-GPU pace (gen takes 130-170s, old gate fired after every generation)" 2>&1 | tail -2
  sleep 25; $K competitions submissions arc-prize-2026-arc-agi-3 2>&1 | head -3 | tail -1
else
  echo "⏸ 机制指标未改善, 未提交 —— 等你看过再定"
fi
