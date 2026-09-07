#!/bin/bash
# upscale 4 vs 8 对照: 3局 × 3passes × 2组, 每局 15 分钟, 组内并发 3
HC=/Users/01450825/Desktop/project/arc-agi-3/probes/20260903/harness_compare
AB=/Users/01450825/Desktop/project/arc-agi-3/probes/20260907/upscale_ab
set -a; source "$HOME/Desktop/2-快件质量与OpenClaw/langfuse-local/.env"; set +a
export OPENROUTER_API_KEY="$AIPLAT_KEY" OPERATION_MODE=OFFLINE
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export PYTHONPATH="$HC/shims"
cd /Users/01450825/Desktop/project/arc-agi-3/reference/fnrep-source-v4/src/ARC3-Inference
for UP in 4 8; do
  echo "[$(date '+%F %T')] ===== 组 upscale=$UP 开始 (3局×3passes, 并发3, 15min/局) ====="
  make interactive CONFIG_PATH=configs/inference.q27-vision-up${UP}.json \
    N_PASSES=3 MAX_RUNTIME_MINUTES=15 CONCURRENT_JOBS=3 RUN_NAME="ab_up${UP}" \
    2>&1 | grep -E "^\[finished\]|^mean score|^median score|^total actions|^runs:|^duration|===== |^Run directory"
  echo "[$(date '+%F %T')] ===== 组 upscale=$UP 结束 ====="
done
echo "[$(date '+%F %T')] ALL DONE"
