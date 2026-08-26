#!/bin/bash
set -a; source "$HOME/Desktop/2-快件质量与OpenClaw/langfuse-local/.env"; set +a
export OPENROUTER_API_KEY="$AIPLAT_KEY" OPERATION_MODE=OFFLINE; export HUB_429_BACKOFF=1; unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy HUB_THINKING_DISABLED
export DUCK_FIELD_GUIDE=$HOME/Desktop/project/arc-agi-3/kaggle_agent/duck/field_guide_en.txt
cd ~/Desktop/project/arc-agi-3/reference/duck-harness/ARC3-Inference
make interactive CONFIG_PATH=configs/inference.hub-q27-ab.json GAME=sb26,r11l N_PASSES=3 CONCURRENT_JOBS=2 MAX_RUNTIME_MINUTES=30 RUN_NAME=q27-ab-guide
echo "[$(date '+%H:%M:%S')] Q27_guide_DONE"
