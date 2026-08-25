#!/bin/bash
# 打新游戏 vc33: dsh 改进版(ctx32k, 提交物同款裁剪+瘦身) × DeepSeek, 400 步 / 40 分钟
set -a; source "$HOME/Desktop/2-快件质量与OpenClaw/langfuse-local/.env"; set +a
export DEEPSEEK_API_KEY="$AIPLAT_KEY" no_proxy=127.0.0.1,localhost NO_PROXY=127.0.0.1,localhost
ROOT=$HOME/Desktop/project/arc-agi-3; O=$ROOT/probes/20260825/play_vc33/dsh; cd $O
sed 's/"--max-actions", "200"/"--max-actions", "400"/' $ROOT/probes/20260825/ab_ctx32k/run_ab2.py > $O/run_one.py
AB_PATCH_SLIM=aiplat-ds-trim-ctx32k.patch.yml AB_STATE_SLIM=1 AB_GAMES=vc33 AB_ARMS=slim AB_SEEDS=1 AB_WALL=2400 AB_PORT0=19600 AB_OUT=$O \
uv run --project $ROOT python $O/run_one.py > $O/run.log 2>&1
echo "[$(date '+%H:%M:%S')] done: $(tail -1 $O/run.log)"; echo VC33_DSH_DONE
