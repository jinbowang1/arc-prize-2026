#!/bin/bash
cd /Users/01450825/Desktop/project/arc-agi-3
PY=./reference/fnrep-source-v4/src/ARC3-Inference/.venv/bin/python
for g in ls20 ft09 cd82 tr87 ar25; do
  for s in 0 1 2; do
    out=$($PY probes/20260907/winframe/verify_bug.py $g $s 8000 2>&1 | grep -E "🎯|GameState.frame|赢的那一刻|差异格子|颜色分布|只有 1 帧")
    if [ -n "$out" ]; then echo "=== $g seed=$s ==="; echo "$out"; fi
  done
done
echo "SWEEP DONE"
