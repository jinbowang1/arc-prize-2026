#!/bin/bash
# 公开集 25 局 A/B: 持久工作台+numpy 开 vs 关。单一变量(闸门/winframe/账本全关)
# g0 = 昨日基线(代码不跨轮·无 numpy·旧提示词) / g1 = 工作台(函数跨轮·notes·numpy)
HC=/Users/01450825/Desktop/project/arc-agi-3/probes/20260903/harness_compare
set -a; source "$HOME/Desktop/2-快件质量与OpenClaw/langfuse-local/.env"; set +a
export OPENROUTER_API_KEY="$AIPLAT_KEY" OPERATION_MODE=OFFLINE HUB_THINKING_DISABLED=1
export ARC3_WINFRAME=0 ARC3_LEDGER=0 ARC3_WM_GATE=0
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
# 🚨 unset 不够: Python 客户端会读 macOS 系统代理(Clash 12639), 请求照走代理并挂死。
# 必须显式 no_proxy=* 才能跳过系统代理。09-08 卡死 40 分钟的根因。
export no_proxy='*' NO_PROXY='*'
export PYTHONPATH="$HC/shims"
cd /Users/01450825/Desktop/project/arc-agi-3/reference/fnrep-source-v4/src/ARC3-Inference
for TAG in g0 g1; do
  export ARC3_WORKBENCH=${TAG#g}
  echo "[$(date '+%F %T')] ===== 组 $TAG (工作台=$ARC3_WORKBENCH) 开跑: 25 局 ====="
  # 全量写盘再提摘要 —— macOS 没有 stdbuf, 管道过滤会被块缓冲憋到结束才出
  FULL=/Users/01450825/Desktop/project/arc-agi-3/probes/20260908/workbench/${TAG}.full.log
  make interactive CONFIG_PATH=configs/inference.wb-${TAG}.json \
    N_PASSES=1 MAX_RUNTIME_MINUTES=12 CONCURRENT_JOBS=6 RUN_NAME="wb_${TAG}" \
    > "$FULL" 2>&1
  grep -E "^\[finished\]|^runs:|^mean score|^duration" "$FULL"
  echo "[$(date '+%F %T')] ===== 组 $TAG 结束 ====="
done
echo "[$(date '+%F %T')] WORKBENCH AB DONE"
