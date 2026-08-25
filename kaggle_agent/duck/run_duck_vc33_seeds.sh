#!/bin/bash
set -a; source "$HOME/Desktop/2-快件质量与OpenClaw/langfuse-local/.env"; set +a
export OPENROUTER_API_KEY="$AIPLAT_KEY" OPERATION_MODE=OFFLINE HUB_THINKING_DISABLED=1; unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
cd ~/Desktop/project/arc-agi-3/reference/duck-harness/ARC3-Inference
make interactive CONFIG_PATH=configs/inference.hub-ds.json GAME=vc33 N_PASSES=2 MAX_RUNTIME_MINUTES=40 CONCURRENT_JOBS=2 RUN_NAME=duck-vc33-seeds
echo "[$(date '+%H:%M:%S')] DUCK_DONE rc=$?"
