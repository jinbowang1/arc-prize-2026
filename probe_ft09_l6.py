"""L6: 点击前后块内容精查(铆钉/色态)。"""
import json
import numpy as np
import arc_agi
from arcengine import GameAction, ActionInput
from solve_ft09_all import clone, raw

arc = arc_agi.Arcade()
env = arc.make("ft09")
f = env.reset()
sols = json.load(open("ft09_solutions.json"))
for seq in sols["seqs"]:
    for (x, y) in seq:
        f = env.step(GameAction.ACTION6, {"x": x, "y": y})
game = env._game
g0 = np.array(f.frame[-1])
CH = ".123456789ABCDEF"

def blk(g, y, x):
    return ["".join(CH[v] for v in g[r, x-2:x+4]) for r in range(y-2, y+4)]

for target in [(16, 14), (24, 14)]:
    y, x = target
    ch = clone(game)
    fr = raw(ch, 6, {"x": x, "y": y})
    g1 = np.array(fr.frame[-1])
    print(f"— click({x},{y})")
    for (yy, xx) in [(8, 14), (16, 14), (24, 14), (16, 6), (16, 22)]:
        b0, b1 = blk(g0, yy, xx), blk(g1, yy, xx)
        mark = " <>" if b0 != b1 else ""
        print(f"  格({yy},{xx}): {' '.join(b0)}  ->  {' '.join(b1)}{mark}")
    # 连点第二次
    fr = raw(ch, 6, {"x": x, "y": y})
    g2 = np.array(fr.frame[-1])
    print(f"  再点一次后 ({y},{x}): {' '.join(blk(g2, y, x))}")
