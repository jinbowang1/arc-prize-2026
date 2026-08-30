#!/bin/bash
# 第九发 55884242 出分轮询
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
while true; do l=$(~/.local/bin/kaggle competitions submissions arc-prize-2026-arc-agi-3 2>&1 | grep 55884242); echo "[$(date '+%H:%M')] $(echo "$l" | grep -o 'SubmissionStatus\.[A-Z]* *[0-9.]*')"
  echo "$l" | grep -q "COMPLETE\|ERROR" && { echo "$l"; break; }; sleep 600; done
