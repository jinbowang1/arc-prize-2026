#!/bin/bash
# 三层框架第一层: DeepSeek 上跑 上下文上限 A/B, 3局×2组×3种子=18路一次并行(DeepSeek 桶额度充裕)
# 等 35b-a3b 的第三批结束再起, 避免 24 路 dsh 同时挤内存
set -a; source "$HOME/Desktop/2-快件质量与OpenClaw/langfuse-local/.env"; set +a
export DEEPSEEK_API_KEY="$AIPLAT_KEY" no_proxy=127.0.0.1,localhost NO_PROXY=127.0.0.1,localhost
ROOT=$HOME/Desktop/project/arc-agi-3; HERE=$ROOT/probes/20260825/ab_ctx32k
until grep -q ALL_DONE $HERE/batches.log; do sleep 20; done
O=$HERE/ds18; mkdir -p $O; cd $O
echo "[$(date '+%H:%M:%S')] ds18 start"
AB_PATCH_CTRL=aiplat-ds-trim.patch.yml AB_PATCH_SLIM=aiplat-ds-trim-ctx32k.patch.yml AB_STATE_CTRL=1 AB_STATE_SLIM=1 \
AB_GAMES=r11l,ft09,ls20 AB_ARMS=ctrl,slim AB_SEEDS=3 AB_WALL=1800 AB_PORT0=19500 AB_OUT=$O \
uv run --project $ROOT python $HERE/run_ab2.py > $O/run.log 2>&1
echo "[$(date '+%H:%M:%S')] ds18 done: $(tail -1 $O/run.log)"; echo DS_DONE
