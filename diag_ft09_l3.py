"""L3: 可编辑格坐标 + 两态内容 + 对称性检验。"""
import copy, json
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
cells = []
for (y, x) in cands:
    ch = clone(game)
    fr = raw(ch, 6, {"x": x, "y": y})
    g1 = np.array(fr.frame[-1])
    if np.any(g1[:63] != g0[:63]):
        d = np.argwhere(g1[:63] != g0[:63])
        c_from = g0[d[0][0], d[0][1]]
        c_to = g1[d[0][0], d[0][1]]
        cells.append((y, x, int(c_from), int(c_to), len(d)))
print("可编辑格 (y, x, 原色, 点后色, 变动px):")
for c in cells:
    print("  ", c)

# 对称检验: 全图(去 HUD 行0-7 cols60-63, 底条63)在 fy/fx/r180 下的不一致对
area = g0[:62].copy()
area[0:8, 60:64] = 4
for name, tr in (("fy", np.flipud), ("fx", np.fliplr), ("r180", lambda m: np.rot90(m, 2))):
    t = tr(area)
    print(f"{name}: 不一致像素 {int(np.sum(t != area))}")
