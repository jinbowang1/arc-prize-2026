#!/bin/bash
# 干净基线重跑: 修好 numpy 开关 bug 之后的 ARC3_WORKBENCH=0
# 等 g1 结束后自动开跑, 与 probes/20260908/workbench/g1 对照
HC=/Users/01450825/Desktop/project/arc-agi-3/probes/20260903/harness_compare
OUT=/Users/01450825/Desktop/project/arc-agi-3/probes/20260908/clean_g0
until ! pgrep -f "inference-taaf-run" >/dev/null 2>&1; do sleep 60; done
echo "[$(date '+%F %T')] g1 已结束, 干净基线开跑"
set -a; source "$HOME/Desktop/2-快件质量与OpenClaw/langfuse-local/.env"; set +a
export OPENROUTER_API_KEY="$AIPLAT_KEY" OPERATION_MODE=OFFLINE HUB_THINKING_DISABLED=1
export ARC3_WINFRAME=0 ARC3_LEDGER=0 ARC3_WM_GATE=0 ARC3_WORKBENCH=0
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export no_proxy='*' NO_PROXY='*'
export PYTHONPATH="$HC/shims"
cd /Users/01450825/Desktop/project/arc-agi-3/reference/fnrep-source-v4/src/ARC3-Inference
make interactive CONFIG_PATH=configs/inference.cleang0.json \
  N_PASSES=1 MAX_RUNTIME_MINUTES=12 CONCURRENT_JOBS=6 RUN_NAME="clean_g0" > "$OUT/full.log" 2>&1
grep -E "^\[finished\]|^runs:|^mean score|^duration" "$OUT/full.log"
echo "[$(date '+%F %T')] CLEAN G0 DONE"
