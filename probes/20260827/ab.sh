#!/bin/bash
# 第一层 A/B 跑一臂。用法: ab.sh <臂名> <开关设置...>
#   例: ab.sh board DUCK_BOARD_IN_PROMPT=1
#       ab.sh ctrl
# 规程见 notes/experiment-protocol-20260827.md —— 关键: 开思考(不设 HUB_THINKING_DISABLED)
set -e
ARM=${1:-}
[ -z "$ARM" ] && { echo "用法: ab.sh <臂名> [VAR=值 ...]"; exit 1; }
shift
set -a; source "$HOME/Desktop/2-快件质量与OpenClaw/langfuse-local/.env"; set +a
export OPENROUTER_API_KEY="$AIPLAT_KEY" OPERATION_MODE=OFFLINE HUB_429_BACKOFF=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
# 一律清空所有实验开关, 再按参数逐个打开 —— 防止上一次实验的残留串味
unset HUB_THINKING_DISABLED DUCK_BOARD_IN_PROMPT DUCK_BOOTSTRAP_PROBE DUCK_FIELD_GUIDE \
      DUCK_FIELD_GUIDE_MODE LOCAL_ANALYZER_TOOL_STEPS LOCAL_ANALYZER_TEMPERATURE \
      LOCAL_ANALYZER_SEED LOCAL_ANALYZER_MAX_OUTPUT ARC3_ANIMATION_AWARENESS ARC3_HARD_NOOP_GUARD
for kv in "$@"; do export "$kv"; done

GAMES=${GAMES:-ft09,r11l,ls20,sb26,cd82,vc33}
PASSES=${PASSES:-3}
MINUTES=${MINUTES:-120}
JOBS=${JOBS:-6}

echo "════════ 臂=$ARM ════════"
echo "  局: $GAMES  遍数: $PASSES  时长: ${MINUTES}min  并发: $JOBS"
echo "  思考: ${HUB_THINKING_DISABLED:-开(未设 HUB_THINKING_DISABLED)}"
echo -n "  本臂开关: "; if [ $# -eq 0 ]; then echo "(无 = 对照臂)"; else echo "$@"; fi
echo "  其余开关已全部清空"
echo "════════════════════════"
cd ~/Desktop/project/arc-agi-3/reference/duck-harness/ARC3-Inference
make interactive CONFIG_PATH=configs/inference.hub-ds-ab.json GAME=$GAMES N_PASSES=$PASSES \
  CONCURRENT_JOBS=$JOBS MAX_RUNTIME_MINUTES=$MINUTES RUN_NAME=ab-$ARM
echo "[$(date '+%H:%M:%S')] DONE $ARM"
