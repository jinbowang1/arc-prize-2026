"""sc25 攻关模式(允许调预算, 与 blind_sc25.py 的盲测口径分开记)。

盲测(blind_sc25.py, 默认预算)的结果已经记在案:
    修指纹前: BFS 扩展 1 节点、深度 0 —— 假穷尽, 1 秒结束
    修指纹后: BFS 扩展 891 / best_first 扩展 2338、深度 9、**h 16 -> 3**, 超时停

h 在降且是超时停的, 所以先加一次预算看它还动不动 —— **判据是"加了算力之后还动
不动", 不是"曾经动过"**(08-12 在 cd82 L3 上据后者预测长跑能破, 实测算力 7.7 倍
h 一格没动)。这一跑就是那个判据的执行。

⚠️预算仍是秒(agent.py 那批还没改成确定性扩展数), 所以本跑不可用于严格对照。
"""
import glob
import json
import sys
import time

from harness.agent import solve_game
from harness.run import verify_replay

GID = "sc25"
mult = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0

meta = json.load(open(glob.glob(f"environment_files/{GID}/*/metadata.json")[0]))
baselines = meta["baseline_actions"]
cfg = {
    "bfs_seconds": 45.0 * mult,
    "best_first_seconds": 120.0 * mult,
    "slot_seconds": 120.0 * mult,
    "max_nodes": int(20000 * mult),
}
print(f"=== sc25 攻关 (预算 x{mult}) ===", flush=True)
print(f"cfg = {cfg}", flush=True)
print(f"人类基准 {baselines} 合计 {sum(baselines)}", flush=True)

t0 = time.time()
log = solve_game(GID, baselines=baselines, cfg=cfg)
print(log.summary(), flush=True)
print(f"总耗时 {time.time()-t0:.0f}s", flush=True)

if log.solution:
    with open(f"{GID}_solutions.json", "w") as f:
        json.dump({"game": GID, "seq": [str(a) for a in log.solution],
                   "baseline": baselines,
                   "per_level_steps": [r.steps for r in log.records if r.solved]},
                  f, ensure_ascii=False, indent=1)
    ok, n, why = verify_replay(GID, log.solution)
    print(f"[verify] 全新环境重放 {n} 步: {'通过' if ok else '未通关'} — {why}", flush=True)
