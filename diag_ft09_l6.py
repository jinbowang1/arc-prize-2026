"""L6 诊断: want 分配 + 点击变动像素规模(判断是否开关型)。"""
import json
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
level = f.levels_completed + 1

cands = blocks(g0)
cells = {}
for (y, x) in cands:
    ch = clone(game)
    fr = raw(ch, 6, {"x": x, "y": y})
    g1 = np.array(fr.frame[-1])
    d = np.argwhere(g1[:63] != g0[:63])
    if len(d):
        rows = sorted({int(r) for r, _ in d})
        cells[(y, x)] = (int(g0[y, x]), len(d), f"行{rows[0]}-{rows[-1]}")
print("可编辑格 (y,x): (质心色, 变动px, 范围)")
for k, v in sorted(cells.items()):
    print("  ", k, v)
ys = sorted({y for y, _ in cells}); xs = sorted({x for _, x in cells})
print(f"格网 y={ys} x={xs}")
