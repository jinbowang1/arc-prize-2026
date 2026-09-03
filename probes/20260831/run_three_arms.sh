#!/bin/bash
# 08-31 三臂串行 (并发54会撞阿里TPM按key×模型分桶, 串行18路最稳):
#   ctrl(对照) → taskbook(A包) → manual(探测说明书)
# 一天改一样落在提交上; 第一层两臂分开验, 互不叠加。
set -e
cd ~/Desktop/project/arc-agi-3
L=probes/20260831
bash $L/ab_aiplat.sh ctrl                        > $L/arm_ctrl.log 2>&1
echo "[$(date '+%H:%M')] ctrl 完成"
bash $L/ab_aiplat.sh taskbook DUCK_TASKBOOK=1    > $L/arm_taskbook.log 2>&1
echo "[$(date '+%H:%M')] taskbook 完成"
bash $L/ab_aiplat.sh manual DUCK_PROBE_MANUAL=1  > $L/arm_manual.log 2>&1
echo "[$(date '+%H:%M')] manual 完成"
echo ALL_ARMS_DONE
