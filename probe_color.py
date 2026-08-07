"""去 (25,29) 的多色物件, 看是否改变钥匙颜色。"""
import numpy as np, solve_l5
from solve_l5 import build, plan
from wm import Percept, energy, load_env, shape_bits, step
from check_color import panel
move, xform = build()
game, f = load_env("solutions_l4.json")
p = Percept(np.array(f.frame[-1]))
solve_l5.LOCK = (25, 29)
r = plan(move, xform, (40, 49), 410, 42, None)
print("到调色板需", len(r[0]), "步")
for a in r[0]:
    f = step(game, a)
g = np.array(f.frame[-1])
print("到达:", p.key(g), "面板:", panel(g), "能量", energy(g))
for i, a in enumerate([3, 4, 3, 4, 1, 2, 1, 2]):
    f = step(game, a)
    if not f.frame:
        print("死亡"); break
    g = np.array(f.frame[-1])
    print(f"  试{i} a{a}: 位置={p.key(g)} 面板={panel(g)} e={energy(g)}")
