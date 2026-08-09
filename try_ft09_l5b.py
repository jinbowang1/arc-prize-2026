"""L5: 点 6-格纹块看反应, 若可变则补齐蓝图要求。"""
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
level = f.levels_completed + 1
g0 = np.array(f.frame[-1])

SIX = [(14, 24), (30, 24), (46, 40)]
for (y, x) in SIX:
    ch = clone(game)
    ring = []
    prev = g0
    for _ in range(4):
        fr = raw(ch, 6, {"x": x, "y": y})
        g = np.array(fr.frame[-1])
        d = np.argwhere(g[:63] != prev[:63])
        blk = g[y - 2:y + 4, x - 2:x + 4]
        ring.append((len(d), sorted(set(blk.flatten().tolist()))))
        prev = g
    print(f"6块({y},{x}) 点击序列: {ring}")
