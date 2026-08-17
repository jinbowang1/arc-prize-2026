"""sc25 攻关第一步: harness 裸跑, 零人工介入。

靶子按元数据选定(keyboard_click / 6 关 / 人类 350 步), **选之前没看过游戏内容**。
避开 sb26/tn36/su15 —— 那三个留作干净盲测靶。

⚠️**这一跑的成绩带墙钟噪声, 不是干净基线。** agent.py 的预算全是秒
(bfs_seconds/slot_seconds/abstract_seconds/best_first_seconds), 正是 08-17
在 canvas 上改掉、但还没推广到主控的那批。机器忙的时候同一份代码会少搜一截,
见 harness/canvas.py 的 Budget 注释与 results/README.md 第四节。
所以这里的"解出/未解出"只能当**方向指示**, 不能拿来跟别的跑做严格对照。
"""
import glob
import json
import sys
import time

from harness.agent import solve_game
from harness.run import verify_replay

GID = sys.argv[1] if len(sys.argv) > 1 else "sc25"

meta = json.load(open(glob.glob(f"environment_files/{GID}/*/metadata.json")[0]))
baselines = meta.get("baseline_actions")
print(f"=== {GID} 裸跑 ===", flush=True)
print(f"人类基准逐关 = {baselines} (合计 {sum(baselines)})", flush=True)
print(f"tags = {meta.get('tags')}", flush=True)

t0 = time.time()
log = solve_game(GID, baselines=baselines)
print(log.summary(), flush=True)
print(f"总耗时 {time.time()-t0:.0f}s", flush=True)

if log.solution:
    with open(f"{GID}_blind_solution.json", "w") as f:
        json.dump({"game": GID, "seq": [str(a) for a in log.solution],
                   "baseline": baselines}, f, ensure_ascii=False, indent=1)
    ok, n, why = verify_replay(GID, log.solution)
    print(f"[verify] 全新环境重放 {n} 步: {'通过' if ok else '未通关'} — {why}", flush=True)
