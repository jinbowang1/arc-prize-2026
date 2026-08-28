#!/bin/bash
# 08-27 第一层 A/B: 开局硬探测(harness 每关自动把每类动作各走一遍) vs 原样。
# 棋盘进上下文那条路已在第一层证伪, 这里确保它是关的。
set -a; source "$HOME/Desktop/2-快件质量与OpenClaw/langfuse-local/.env"; set +a
export OPENROUTER_API_KEY="$AIPLAT_KEY" OPERATION_MODE=OFFLINE HUB_THINKING_DISABLED=1 HUB_429_BACKOFF=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy DUCK_BOARD_IN_PROMPT
cd ~/Desktop/project/arc-agi-3/reference/duck-harness/ARC3-Inference
ARM=$1
if [ "$ARM" = "probe" ]; then export DUCK_BOOTSTRAP_PROBE=1; else unset DUCK_BOOTSTRAP_PROBE; fi
echo "[$(date '+%H:%M:%S')] 臂=$ARM DUCK_BOOTSTRAP_PROBE=${DUCK_BOOTSTRAP_PROBE:-unset} DUCK_BOARD_IN_PROMPT=${DUCK_BOARD_IN_PROMPT:-unset}"
make interactive CONFIG_PATH=configs/inference.hub-ds-ab.json GAME=ft09,r11l,ls20 N_PASSES=2 \
  CONCURRENT_JOBS=6 MAX_RUNTIME_MINUTES=30 RUN_NAME=probe-ab-$ARM
echo "[$(date '+%H:%M:%S')] DONE $ARM"
