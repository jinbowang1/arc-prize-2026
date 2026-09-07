#!/bin/bash
# 探索账本 A/B: ARC3_LEDGER 开关分组; ARC3_WINFRAME=0 保证单一变量
HC=/Users/01450825/Desktop/project/arc-agi-3/probes/20260903/harness_compare
set -a; source "$HOME/Desktop/2-快件质量与OpenClaw/langfuse-local/.env"; set +a
export OPENROUTER_API_KEY="$AIPLAT_KEY" OPERATION_MODE=OFFLINE HUB_THINKING_DISABLED=1
export ARC3_WINFRAME=0
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export PYTHONPATH="$HC/shims"
cd /Users/01450825/Desktop/project/arc-agi-3/reference/fnrep-source-v4/src/ARC3-Inference
for TAG in lg0 lg1; do
  export ARC3_LEDGER=${TAG#lg}
  echo "[$(date '+%F %T')] ===== 组 $TAG (ARC3_LEDGER=$ARC3_LEDGER) 开跑 ====="
  make interactive CONFIG_PATH=configs/inference.ledger-${TAG}.json \
    N_PASSES=3 MAX_RUNTIME_MINUTES=12 CONCURRENT_JOBS=6 RUN_NAME="ab_${TAG}" \
    2>&1 | grep -E "^\[finished\]|^mean score|^runs:|^duration"
  echo "[$(date '+%F %T')] ===== 组 $TAG 结束 ====="
done
echo "[$(date '+%F %T')] LEDGER AB DONE"
