#!/bin/bash
set -a; source "$HOME/Desktop/2-快件质量与OpenClaw/langfuse-local/.env"; set +a
export DEEPSEEK_API_KEY="$AIPLAT_KEY" no_proxy=127.0.0.1,localhost NO_PROXY=127.0.0.1,localhost
ROOT=$HOME/Desktop/project/arc-agi-3; O=$ROOT/probes/20260825/play_vc33/dsh_seeds; mkdir -p $O; cd $O
AB_PATCH_SLIM=aiplat-ds-trim-ctx32k.patch.yml AB_STATE_SLIM=1 AB_GAMES=vc33 AB_ARMS=slim AB_SEEDS=2 AB_WALL=2400 AB_PORT0=19650 AB_OUT=$O \
uv run --project $ROOT python $ROOT/probes/20260825/play_vc33/dsh/run_one.py > $O/run.log 2>&1
echo "[$(date '+%H:%M:%S')] done: $(tail -1 $O/run.log)"; echo DSH_SEEDS_DONE
