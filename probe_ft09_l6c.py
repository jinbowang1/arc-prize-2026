"""L6: 22 格全量影响矩阵(每个点击翻转哪些格)。"""
import json, pickle
import numpy as np
import arc_agi
from arcengine import GameAction, ActionInput
from solve_ft09_all import blocks, clone, raw

arc = arc_agi.Arcade()
env = arc.make("ft09")
f = env.reset()
sols = json.load(open("ft09_solutions.json"))
for seq in sols["seqs"]:
    for (x, y) in seq:
        f = env.step(GameAction.ACTION6, {"x": x, "y": y})
game = env._game
g0 = np.array(f.frame[-1])

cands = blocks(g0)
cells = []
for (y, x) in cands:
    ch = clone(game)
    fr = raw(ch, 6, {"x": x, "y": y})
    if np.any(np.array(fr.frame[-1])[:63] != g0[:63]):
        cells.append((y, x))
cells.sort()

influence = {}
for (y, x) in cells:
    ch = clone(game)
    fr = raw(ch, 6, {"x": x, "y": y})
    g1 = np.array(fr.frame[-1])
    flipped = [(yy, xx) for (yy, xx) in cells if g1[yy, xx] != g0[yy, xx]]
    influence[(y, x)] = flipped
    model = [(y, x)] + ([(y - 8, x)] if (y - 8, x) in cells else [])
    tag = "✓self+up" if sorted(flipped) == sorted(model) else f"⚠️实际={flipped}"
    print(f"click({y},{x}): {tag}")
pickle.dump({"cells": cells, "influence": influence}, open("ft09_l6_influence.pkl", "wb"))
