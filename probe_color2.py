"""摸清调色板的颜色循环, 目标色 8。"""
import numpy as np, solve_l5
from solve_l5 import build, plan
from wm import Percept, energy, load_env, step
from check_color import panel
move, xform = build()
game, f = load_env("solutions_l4.json")
p = Percept(np.array(f.frame[-1]))
solve_l5.LOCK = (25, 29)
r = plan(move, xform, (40, 49), 410, 42, None)
for a in r[0]:
    f = step(game, a)
g = np.array(f.frame[-1])
cols = [panel(g)[1]]
print("到达调色板, 色", cols[0], "能量", energy(g))
for i in range(10):
    f = step(game, 4)          # 出
    if not f.frame: print("死亡@出", i); break
    f = step(game, 3)          # 进
    if not f.frame: print("死亡@进", i); break
    g = np.array(f.frame[-1])
    sh, c = panel(g)
    cols.append(c)
    print(f"  第{i+2}次进入: 色={c} 形状={sh} e={energy(g)}")
    if c == 8:
        print("*** 拿到目标色 8!")
        break
print("颜色序列:", cols)
