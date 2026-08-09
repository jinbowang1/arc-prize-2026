import json
import numpy as np
import arc_agi
from arcengine import GameAction

arc = arc_agi.Arcade()
env = arc.make("ft09")
f = env.reset()
sols = json.load(open("ft09_solutions.json"))
for seq in sols["seqs"]:
    for (x, y) in seq:
        f = env.step(GameAction.ACTION6, {"x": x, "y": y})
g = np.array(f.frame[-1])
CH = ".123456789ABCDEF"
print(f"levels={f.levels_completed}")
for r in range(64):
    print(f"{r:>3} " + "".join(CH[v] for v in g[r]))
