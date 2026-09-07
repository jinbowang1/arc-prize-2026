#!/bin/bash
# 世界模型闸门实验: 前 N 步自由, 之后必须有通过回放验证的 world_model 才能继续动作
HC=/Users/01450825/Desktop/project/arc-agi-3/probes/20260903/harness_compare
set -a; source "$HOME/Desktop/2-快件质量与OpenClaw/langfuse-local/.env"; set +a
export OPENROUTER_API_KEY="$AIPLAT_KEY" OPERATION_MODE=OFFLINE HUB_THINKING_DISABLED=1
export ARC3_WINFRAME=0 ARC3_LEDGER=0
export ARC3_WM_GATE=${ARC3_WM_GATE:-1}
export ARC3_WM_FREE_STEPS=${ARC3_WM_FREE_STEPS:-40}
export ARC3_WM_MIN_CHECKED=${ARC3_WM_MIN_CHECKED:-8}
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export PYTHONPATH="$HC/shims"
cd /Users/01450825/Desktop/project/arc-agi-3/reference/fnrep-source-v4/src/ARC3-Inference
make interactive CONFIG_PATH=configs/inference.ds-wf0.json GAME="${1:-ft09}" N_PASSES=1 \
  MAX_RUNTIME_MINUTES=${2:-15} CONCURRENT_JOBS=1 RUN_NAME="${3:-wm_gate}"
