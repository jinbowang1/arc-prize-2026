#!/bin/bash
# expect 冒烟/AB: ARC3_WINFRAME=0 保证单一变量(只测 expect)
HC=/Users/01450825/Desktop/project/arc-agi-3/probes/20260903/harness_compare
set -a; source "$HOME/Desktop/2-快件质量与OpenClaw/langfuse-local/.env"; set +a
export OPENROUTER_API_KEY="$AIPLAT_KEY" OPERATION_MODE=OFFLINE HUB_THINKING_DISABLED=1
export ARC3_WINFRAME=0
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export PYTHONPATH="$HC/shims"
cd /Users/01450825/Desktop/project/arc-agi-3/reference/fnrep-source-v4/src/ARC3-Inference
CFG=${CFG:-configs/inference.ds-wf0.json}
make interactive CONFIG_PATH=$CFG GAME="${1:-ft09}" N_PASSES=${2:-1} \
  MAX_RUNTIME_MINUTES=${3:-10} CONCURRENT_JOBS=${4:-1} RUN_NAME="${5:-expect_smoke}"
