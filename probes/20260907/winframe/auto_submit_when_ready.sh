#!/bin/bash
# kernel 跑完 -> 自动提交 winframe。若到 06:00 仍未就绪, 改投 5.49 原版保住名额。
export http_proxy=http://127.0.0.1:6789 https_proxy=http://127.0.0.1:6789
KG=~/.local/bin/kaggle
PY=~/.local/share/uv/tools/kaggle/bin/python
BASE=~/Desktop/project/arc-agi-3/probes/20260907/winframe
LOG=$BASE/auto_submit.log
say(){ echo "[$(date '+%m-%d %H:%M')] $*" | tee -a $LOG; }

say "守候开始: 等 arc3-v13-winframe-run 跑完后自动提交"
while true; do
  st=$($KG kernels status jinbowang1/arc3-v13-winframe-run 2>&1 | head -1)
  case "$st" in
    *COMPLETE*)
      say "kernel 跑完 -> 提交 winframe"
      $PY $BASE/submit_sdk.py 2>&1 | tail -6 | tee -a $LOG
      exit 0 ;;
    *ERROR*|*CANCEL*)
      say "🚨 kernel 失败($st) -> 不提交 winframe, 保留名额待人工决定"
      exit 1 ;;
  esac
  # 兜底: 北京时间 06:00 后仍未就绪, 投 5.49 原版(它已跑通, 有输出文件)
  h=$(date '+%H')
  if [ "$h" -ge 6 ] && [ "$h" -lt 8 ]; then
    say "已到 06:00 且 kernel 仍未就绪($st) -> 改投 5.49 原版保住名额"
    WF_SLUG=arc3-fnrep-run WF_VER=4 \
    WF_MSG="5.49 v4 bundle unchanged. The winframe build could not be submitted in time (its notebook was still queued and had produced no output file), so keeping the known-good configuration for this slot." \
    $PY $BASE/submit_sdk.py 2>&1 | tail -6 | tee -a $LOG
    exit 0
  fi
  sleep 300
done
