"""跑 ReAct + Plan-Execute 主循环。

用法: run_agent.py <game_id> [关卡上限]
"""
import glob
import json
import sys
import time

from harness.agent import solve_game
from harness.run import verify_replay

GID = sys.argv[1] if len(sys.argv) > 1 else "r11l"

meta = json.load(open(glob.glob(f"environment_files/{GID}/*/metadata.json")[0]))
baselines = meta.get("baseline_actions")
print(f"人类基准逐关 = {baselines} (合计 {sum(baselines)})", flush=True)

t0 = time.time()
log = solve_game(GID, baselines=baselines)
print("\n" + log.summary(), flush=True)
print(f"总耗时 {time.time()-t0:.0f}s", flush=True)

if log.solution:
    ok, n, why = verify_replay(GID, log.solution)
    print(f"[verify] 全新环境重放 {n} 步: {'通过' if ok else '未通关'} — {why}", flush=True)
