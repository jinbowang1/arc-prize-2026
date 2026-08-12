"""r11l 盲测:harness 裸跑,零人工介入。

靶子按元数据选定(纯 click / 6 关 / 人类 233 步),未预看游戏内容。
cd82 已被调试污染,只有干净靶子上的第一遍才算盲测分数。
"""
import json
import glob
import sys
import time

from harness.run import solve_game, verify_replay

GID = sys.argv[1] if len(sys.argv) > 1 else "r11l"

meta = json.load(open(glob.glob(f"environment_files/{GID}/*/metadata.json")[0]))
baselines = meta.get("baseline_actions")
print(f"人类基准逐关 = {baselines} (合计 {sum(baselines)})", flush=True)

t0 = time.time()
log = solve_game(GID, baselines=baselines,
                 max_nodes=int(sys.argv[2]) if len(sys.argv) > 2 else 20000,
                 bfs_seconds=float(sys.argv[3]) if len(sys.argv) > 3 else 90.0,
                 ask_human=None)
print(log.summary(), flush=True)
print(f"总耗时 {time.time()-t0:.0f}s", flush=True)

if log.solution:
    ok, n, why = verify_replay(GID, log.solution)
    print(f"[verify] 全新环境重放 {n} 步: {'通过' if ok else '未通关'} — {why}", flush=True)
