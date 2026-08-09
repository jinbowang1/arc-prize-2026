import json
import numpy as np
import arc_agi
from arcengine import GameAction

ACTS = {i: getattr(GameAction, f"ACTION{i}") for i in range(1, 8)}
arc = arc_agi.Arcade()
env = arc.make("tr87")
f = env.reset()
sols = json.load(open("tr87_solutions.json"))
for seq in sols["seqs"]:
    for a in seq:
        f = env.step(ACTS[a])
g = np.array(f.frame[-1])
CH = ".123456789ABCDEF"
print(f"levels={f.levels_completed} 颜色: {dict(zip(*[x.tolist() for x in np.unique(g, return_counts=True)]))}")
for r in range(64):
    print(f"{r:>3}  " + "".join(CH[v] for v in g[r]))
