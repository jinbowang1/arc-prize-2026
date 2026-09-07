#!/bin/bash
# run_up.sh <upscale> <game> <run_name> <max_runtime_minutes>
UP=$1; GAME=$2; RUN=$3; MRM=${4:-15}
HC=/Users/01450825/Desktop/project/arc-agi-3/probes/20260903/harness_compare
set -a; source "$HOME/Desktop/2-快件质量与OpenClaw/langfuse-local/.env"; set +a
export OPENROUTER_API_KEY="$AIPLAT_KEY" OPERATION_MODE=OFFLINE
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export PYTHONPATH="$HC/shims"
cd /Users/01450825/Desktop/project/arc-agi-3/reference/fnrep-source-v4/src/ARC3-Inference
echo "[$(date '+%F %T')] START upscale=$UP game=$GAME run=$RUN mrm=$MRM"
make interactive CONFIG_PATH=configs/inference.q27-vision-up${UP}.json \
  GAME="$GAME" N_PASSES=1 MAX_RUNTIME_MINUTES=$MRM CONCURRENT_JOBS=1 RUN_NAME="$RUN"
echo "[$(date '+%F %T')] DONE upscale=$UP game=$GAME rc=$?"
