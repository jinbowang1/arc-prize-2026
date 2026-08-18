"""那 5 个"复活"的点击在画布涂过之后改了什么? 显示器真的是静态的吗?"""
import json
import numpy as np
from harness.env import Action, Game
from harness.run import _parse

PAL = " .:-=+*#%@$&XO<>"
sol = json.load(open("sc25_solutions.json"))
game, obs = Game.make("sc25")
for t in sol["seq"]:
    obs = game.act(_parse(t))
R, C = [49, 54, 59], [24, 29, 34]

# 造"画布涂过"的现场
n = game.fork()
for i in range(3):
    n.act(Action.click(C[1] + 1, R[i] + 1))
base = np.array(n._grid())
print("画布涂过后的现场已就绪\n")

REVIVED = [(11,55), (12,55), (15,51), (15,54), (15,57)]
for (x, y) in REVIVED:
    a = Action.click(x, y)
    o = n.effect(a)
    g = np.array(o.grid)
    d = np.argwhere(g != base)
    rows = sorted({int(r) for r,_ in d}); cols = sorted({int(c) for _,c in d})
    print(f"{repr(a):<14} 改 {len(d):>3} 格 | 行{rows[0]}..{rows[-1]} 列{cols[0]}..{cols[-1]} "
          f"| 颜色 {sorted({int(base[r,c]) for r,c in d})} -> {sorted({int(g[r,c]) for r,c in d})}")

# 显示器区域前后对比
a = Action.click(*REVIVED[0])
o = n.effect(a)
g = np.array(o.grid)
print(f"\n显示器区 行46..62 列8..26  (点 {repr(a)} 之前 -> 之后):")
for r in range(46, 63):
    b = "".join(PAL[v % 16] for v in base[r, 8:27])
    c = "".join(PAL[v % 16] for v in g[r, 8:27])
    mark = "  <>" if b != c else ""
    print(f"  {r:>3} {b}   {c}{mark}")
