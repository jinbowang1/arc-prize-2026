#!/bin/bash
# A/B #1 上下文上限: ctrl=aiplat-qwen-trim(128k) vs slim=aiplat-qwen-trim-ctx32k(32k); 两组都用 /state 瘦身
# 3 局 × 2 组 × 3 种子 = 18 路, 按种子分 3 批串行(hub 按每分钟 token 限流, 6 路一批)
set -a; source "$HOME/Desktop/2-快件质量与OpenClaw/langfuse-local/.env"; set +a
export DEEPSEEK_API_KEY="$AIPLAT_KEY" no_proxy=127.0.0.1,localhost NO_PROXY=127.0.0.1,localhost
ROOT=$HOME/Desktop/project/arc-agi-3; HERE=$ROOT/probes/20260825/ab_ctx32k
for sd in 0 1 2; do
  O=$HERE/batch$sd; mkdir -p $O; cd $O
  echo "[$(date '+%H:%M:%S')] batch$sd start"
  AB_PATCH_CTRL=aiplat-qwen-trim.patch.yml AB_PATCH_SLIM=aiplat-qwen-trim-ctx32k.patch.yml AB_STATE_CTRL=1 AB_STATE_SLIM=1 \
  AB_GAMES=r11l,ft09,ls20 AB_ARMS=ctrl,slim AB_SEEDS=1 AB_WALL=1800 AB_PORT0=$((19400+sd*10)) AB_OUT=$O \
  uv run --project $ROOT python $HERE/run_ab2.py > $O/run.log 2>&1
  echo "[$(date '+%H:%M:%S')] batch$sd done: $(tail -1 $O/run.log)"
done
echo ALL_DONE
