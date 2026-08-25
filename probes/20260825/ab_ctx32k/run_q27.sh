#!/bin/bash
# 三层框架第二层: qwen3.6-27b 验证 上下文上限; ls20+r11l × 2组 × 3种子 = 12路, 该模型桶约65万token/分只够3路, 分4批串行
set -a; source "$HOME/Desktop/2-快件质量与OpenClaw/langfuse-local/.env"; set +a
export DEEPSEEK_API_KEY="$AIPLAT_KEY" no_proxy=127.0.0.1,localhost NO_PROXY=127.0.0.1,localhost
ROOT=$HOME/Desktop/project/arc-agi-3; HERE=$ROOT/probes/20260825/ab_ctx32k
i=0
for g in ls20 r11l; do for arm in ctrl slim; do
  O=$HERE/q27_${g}_${arm}; mkdir -p $O; cd $O
  echo "[$(date '+%H:%M:%S')] q27 $g/$arm ×3种子 start"
  AB_PATCH_CTRL=aiplat-q27-trim.patch.yml AB_PATCH_SLIM=aiplat-q27-trim-ctx32k.patch.yml AB_STATE_CTRL=1 AB_STATE_SLIM=1 \
  AB_GAMES=$g AB_ARMS=$arm AB_SEEDS=3 AB_WALL=1800 AB_PORT0=$((19700+i*10)) AB_OUT=$O \
  uv run --project $ROOT python $HERE/run_ab2.py > $O/run.log 2>&1
  echo "[$(date '+%H:%M:%S')] done: $(tail -1 $O/run.log)"; i=$((i+1))
done; done
echo Q27_DONE
