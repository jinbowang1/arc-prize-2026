#!/bin/bash
# run_side.sh <duck|son> <game> <run_name> <max_runtime_minutes> [n_passes] [concurrent_jobs]
SIDE=$1; GAME=$2; RUN=$3; MRM=$4; NP=${5:-1}; CJ=${6:-1}
HC=/Users/01450825/Desktop/project/arc-agi-3/probes/20260903/harness_compare
set -a; source "$HOME/Desktop/2-快件质量与OpenClaw/langfuse-local/.env"; set +a
export OPENROUTER_API_KEY="$AIPLAT_KEY" OPERATION_MODE=OFFLINE HUB_THINKING_DISABLED=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export PYTHONPATH="$HC/shims"          # re_arc shim (Son 的 run.py 顶层要 import; duck 用不到, 无副作用)
if [ "$SIDE" = "duck" ]; then
  cd /Users/01450825/Desktop/project/arc-agi-3/reference/duck-harness/ARC3-Inference
else
  cd /Users/01450825/Desktop/project/arc-agi-3/reference/fnrep-source-v4/src/ARC3-Inference
fi
echo "[$(date '+%F %T')] START side=$SIDE game=$GAME run=$RUN mrm=$MRM n_passes=$NP cj=$CJ cwd=$(pwd)"
if [ "$GAME" = "-" ]; then
  # 不传 GAME, 让 Makefile 从 config 的 environment.games 读整张局表(JSON 数组)
  make interactive CONFIG_PATH=configs/inference.hub-ds-cmp.json \
    N_PASSES=$NP MAX_RUNTIME_MINUTES=$MRM CONCURRENT_JOBS=$CJ RUN_NAME="$RUN"
else
  make interactive CONFIG_PATH=configs/inference.hub-ds-cmp.json GAME="$GAME" \
    N_PASSES=$NP MAX_RUNTIME_MINUTES=$MRM CONCURRENT_JOBS=$CJ RUN_NAME="$RUN"
fi
rc=$?
echo "[$(date '+%F %T')] DONE side=$SIDE game=$GAME rc=$rc"
