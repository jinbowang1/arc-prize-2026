#!/bin/bash
# 第一层 A/B —— aiplat 通道 (08-31: 工作日走 llm-model-hub-apis, 模型 deepseek-v4-flash, 免充值)。
# 用法: ab_aiplat.sh <臂名> [VAR=值 ...]   例: ab_aiplat.sh taskbook DUCK_TASKBOOK=1
# 传开关一律逐个写实参 (zsh 不拆分未加引号的 $var, 08-30 两臂白跑的教训)。
set -e
ARM=${1:?用法: ab_aiplat.sh <臂名> [VAR=值 ...]}; shift
set -a; source "$HOME/Desktop/2-快件质量与OpenClaw/langfuse-local/.env"; set +a
export OPENROUTER_API_KEY="$AIPLAT_KEY" OPERATION_MODE=OFFLINE HUB_429_BACKOFF=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
unset HUB_THINKING_DISABLED DUCK_BOARD_IN_PROMPT DUCK_BOOTSTRAP_PROBE DUCK_FIELD_GUIDE DUCK_FIELD_GUIDE_MODE \
      DUCK_CAUSAL_TABLE DUCK_PERSIST_DEFS DUCK_DIVISION_OF_LABOUR DUCK_TASKBOOK DUCK_PROBE_MANUAL \
      LOCAL_ANALYZER_TOOL_STEPS LOCAL_ANALYZER_TEMPERATURE LOCAL_ANALYZER_SEED LOCAL_ANALYZER_MAX_OUTPUT \
      ARC3_ANIMATION_AWARENESS ARC3_HARD_NOOP_GUARD
for kv in "$@"; do export "$kv"; done
GAMES=${GAMES:-ft09,r11l,ls20,sb26,cd82,vc33}; PASSES=${PASSES:-3}; MINUTES=${MINUTES:-45}; JOBS=${JOBS:-18}
echo "════════ 第一层(aiplat) 臂=$ARM ════════"
echo "  模型: aliyun/deepseek-v4-flash-0731@llm-model-hub-apis | 局: $GAMES | ${PASSES}遍 ${MINUTES}min 并发$JOBS | 思考: 开"
echo -n "  本臂开关: "; if [ $# -eq 0 ]; then echo "(无 = 对照臂)"; else echo "$@"; fi
echo "════════════════════════"
cd ~/Desktop/project/arc-agi-3/reference/duck-harness/ARC3-Inference
make interactive CONFIG_PATH=configs/inference.aiplat-ab.json GAME=$GAMES N_PASSES=$PASSES \
  CONCURRENT_JOBS=$JOBS MAX_RUNTIME_MINUTES=$MINUTES RUN_NAME=ab-$ARM
echo "[$(date '+%H:%M:%S')] DONE $ARM"
