#!/bin/bash
# 提交后盯状态: PENDING -> COMPLETE/ERROR。每 10 分钟一次, 状态变化才输出。
export http_proxy=http://127.0.0.1:6789 https_proxy=http://127.0.0.1:6789
COMP=arc-prize-2026-arc-agi-3
prev=""
while true; do
  line=$(kaggle competitions submissions $COMP 2>/dev/null | sed -n '3p')
  st=$(echo "$line" | grep -oE "SubmissionStatus\.[A-Z]+")
  sc=$(echo "$line" | awk '{print $NF}')
  cur="$st $sc"
  if [ "$cur" != "$prev" ]; then
    echo "[$(date '+%m-%d %H:%M')] $st  score=$sc"
    prev="$cur"
  fi
  case "$st" in
    *COMPLETE*) echo "[出分] $line" | cut -c1-160; break ;;
    *ERROR*)    echo "[失败] $line" | cut -c1-160; break ;;
  esac
  sleep 600
done
