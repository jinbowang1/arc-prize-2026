#!/bin/bash
# 第二层 A/B (qwen3.6-27b + 开图 + 开思考, 最接近赛场的一层)。
# 用法: ab2.sh <臂名> [VAR=值 ...]
# 🚨 q27 按 API key×模型 分桶 TPM 约 65 万/分, 并发别开太大; 两臂并行时每臂 ≤12。
set -e
ARM=${1:-}
[ -z "$ARM" ] && { echo "用法: ab2.sh <臂名> [VAR=值 ...]"; exit 1; }
shift
set -a; source "$HOME/Desktop/2-快件质量与OpenClaw/langfuse-local/.env"; set +a
export OPENROUTER_API_KEY="$AIPLAT_KEY" OPERATION_MODE=OFFLINE HUB_429_BACKOFF=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
unset HUB_THINKING_DISABLED DUCK_BOARD_IN_PROMPT DUCK_BOOTSTRAP_PROBE DUCK_FIELD_GUIDE \
      DUCK_FIELD_GUIDE_MODE DUCK_CAUSAL_TABLE DUCK_PERSIST_DEFS DUCK_DIVISION_OF_LABOUR \
      LOCAL_ANALYZER_TOOL_STEPS LOCAL_ANALYZER_TEMPERATURE LOCAL_ANALYZER_SEED \
      LOCAL_ANALYZER_MAX_OUTPUT ARC3_ANIMATION_AWARENESS ARC3_HARD_NOOP_GUARD
for kv in "$@"; do export "$kv"; done
GAMES=${GAMES:-ft09,r11l,ls20,sb26,cd82,vc33}
PASSES=${PASSES:-2}; MINUTES=${MINUTES:-45}; JOBS=${JOBS:-12}
echo "════════ 第二层 臂=$ARM ════════"
echo "  模型: qwen3.6-27b + 开图 | 局: $GAMES | ${PASSES}遍 ${MINUTES}min 并发$JOBS"
echo -n "  本臂开关: "; if [ $# -eq 0 ]; then echo "(无 = 对照臂)"; else echo "$@"; fi
echo "════════════════════════════"
cd ~/Desktop/project/arc-agi-3/reference/duck-harness/ARC3-Inference
make interactive CONFIG_PATH=configs/inference.hub-q27-img.json GAME=$GAMES N_PASSES=$PASSES \
  CONCURRENT_JOBS=$JOBS MAX_RUNTIME_MINUTES=$MINUTES RUN_NAME=l2-$ARM
echo "[$(date '+%H:%M:%S')] DONE $ARM"
