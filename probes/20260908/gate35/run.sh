#!/bin/bash
# 公开集 25 局 A/B: 世界模型闸门 开 vs 关。单一变量(winframe/账本都关)
HC=/Users/01450825/Desktop/project/arc-agi-3/probes/20260903/harness_compare
set -a; source "$HOME/Desktop/2-快件质量与OpenClaw/langfuse-local/.env"; set +a
export OPENROUTER_API_KEY="$AIPLAT_KEY" OPERATION_MODE=OFFLINE HUB_THINKING_DISABLED=1
export ARC3_WINFRAME=0 ARC3_LEDGER=0
export ARC3_WM_FREE_STEPS=${ARC3_WM_FREE_STEPS:-35}
export ARC3_WM_MIN_CHECKED=${ARC3_WM_MIN_CHECKED:-5}
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
# 🚨 unset 不够: Python 客户端会读 macOS 系统代理(Clash 12639), 请求照走代理并挂死。
# 必须显式 no_proxy=* 才能让 httpx/requests 跳过系统代理。09-08 卡死 40 分钟的根因。
export no_proxy='*' NO_PROXY='*'
export PYTHONPATH="$HC/shims"
cd /Users/01450825/Desktop/project/arc-agi-3/reference/fnrep-source-v4/src/ARC3-Inference
for TAG in g0 g1; do
  export ARC3_WM_GATE=${TAG#g}
  echo "[$(date '+%F %T')] ===== 组 $TAG (闸门=$ARC3_WM_GATE) 开跑: 25 局 ====="
  make interactive CONFIG_PATH=configs/inference.gate35-${TAG}.json \
    N_PASSES=1 MAX_RUNTIME_MINUTES=12 CONCURRENT_JOBS=6 RUN_NAME="gate35_${TAG}" \
    2>&1 | grep -E "^\[finished\]|^runs:|^mean score|^duration"
  echo "[$(date '+%F %T')] ===== 组 $TAG 结束 ====="
done
echo "[$(date '+%F %T')] GATE25 DONE"
