#!/bin/bash
# 归因表 A/B。等前一组跑完自动接上, 两臂并行。
# 规程: notes/experiment-protocol-20260827.md (开思考 / 每臂≥400轮 / 判据看机制指标)
cd ~/Desktop/project/arc-agi-3
echo "[$(date '+%H:%M:%S')] 等前一组 A/B 跑完..."
while [ "$(grep -l DONE probes/20260827/r_ctrl.log probes/20260827/r_board.log 2>/dev/null | wc -l | tr -d ' ')" != "2" ]; do sleep 60; done
echo "[$(date '+%H:%M:%S')] 前一组已完成, 起归因表两臂"
nohup bash probes/20260827/ab.sh cz-ctrl            > probes/20260827/c_ctrl.log 2>&1 &
sleep 10
nohup bash probes/20260827/ab.sh cz-tbl DUCK_CAUSAL_TABLE=1 > probes/20260827/c_tbl.log 2>&1 &
sleep 30
echo "── 对照臂 ──"; head -6 probes/20260827/c_ctrl.log
echo "── 实验臂 ──"; head -6 probes/20260827/c_tbl.log
