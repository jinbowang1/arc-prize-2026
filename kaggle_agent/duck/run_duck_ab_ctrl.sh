#!/bin/bash
set -a; source "$HOME/Desktop/2-快件质量与OpenClaw/langfuse-local/.env"; set +a
export OPENROUTER_API_KEY="$AIPLAT_KEY" OPERATION_MODE=OFFLINE HUB_THINKING_DISABLED=1; unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

cd ~/Desktop/project/arc-agi-3/reference/duck-harness/ARC3-Inference
make interactive CONFIG_PATH=configs/inference.hub-ds-ab.json GAME=ft09,r11l,ls20 N_PASSES=2 CONCURRENT_JOBS=6 MAX_RUNTIME_MINUTES=30 RUN_NAME=guide-ab-ctrl
echo "[$(date '+%H:%M:%S')] AB_ctrl_DONE"
