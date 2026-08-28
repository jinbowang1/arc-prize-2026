#!/bin/bash
# 08-27 第一层(DeepSeek) A/B: 棋盘进上下文 vs 原样。两臂唯一差别是 DUCK_BOARD_IN_PROMPT。
set -a; source "$HOME/Desktop/2-快件质量与OpenClaw/langfuse-local/.env"; set +a
export OPENROUTER_API_KEY="$AIPLAT_KEY" OPERATION_MODE=OFFLINE HUB_THINKING_DISABLED=1 HUB_429_BACKOFF=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
cd ~/Desktop/project/arc-agi-3/reference/duck-harness/ARC3-Inference
ARM=$1   # board | ctrl
if [ "${ARM#board}" != "$ARM" ]; then export DUCK_BOARD_IN_PROMPT=1; else unset DUCK_BOARD_IN_PROMPT; fi
echo "[$(date '+%H:%M:%S')] 臂=$ARM DUCK_BOARD_IN_PROMPT=${DUCK_BOARD_IN_PROMPT:-unset}"
make interactive CONFIG_PATH=configs/inference.hub-ds-ab.json GAME=ft09,r11l,ls20 N_PASSES=2 \
  CONCURRENT_JOBS=6 MAX_RUNTIME_MINUTES=30 RUN_NAME=board-ab-$ARM
echo "[$(date '+%H:%M:%S')] DONE $ARM"
