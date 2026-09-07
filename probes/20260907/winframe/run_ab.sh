#!/bin/bash
# winframe A/B: 同一份代码, 用 ARC3_WINFRAME 开关分组
HC=/Users/01450825/Desktop/project/arc-agi-3/probes/20260903/harness_compare
set -a; source "$HOME/Desktop/2-快件质量与OpenClaw/langfuse-local/.env"; set +a
export OPENROUTER_API_KEY="$AIPLAT_KEY" OPERATION_MODE=OFFLINE HUB_THINKING_DISABLED=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export PYTHONPATH="$HC/shims"
cd /Users/01450825/Desktop/project/arc-agi-3/reference/fnrep-source-v4/src/ARC3-Inference
for TAG in wf0 wf1; do
  export ARC3_WINFRAME=${TAG#wf}
  echo "[$(date '+%F %T')] ===== 组 $TAG (ARC3_WINFRAME=$ARC3_WINFRAME) 开跑: 4局×3passes 并发6 ====="
  make interactive CONFIG_PATH=configs/inference.ds-${TAG}.json \
    N_PASSES=3 MAX_RUNTIME_MINUTES=12 CONCURRENT_JOBS=6 RUN_NAME="ab_${TAG}" \
    2>&1 | grep -E "^\[finished\]|^mean score|^runs:|^duration|^total actions|^Run directory"
  echo "[$(date '+%F %T')] ===== 组 $TAG 结束 ====="
done
echo "[$(date '+%F %T')] WINFRAME AB DONE"
