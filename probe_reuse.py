"""补给是一次性还是可重复?来回踩两次实测。"""
import numpy as np
from plan_l6 import load, plan, ROTS, XFORMS, PALETTE
from wm import Percept, energy, load_env, panel_color, shape_bits, step
MECH = ROTS | XFORMS | PALETTE
move, fam, pickups = load()
game, f = load_env("solutions_l5.json")
p = Percept(np.array(f.frame[-1]))
def st():
    g = np.array(f.frame[-1]); return p.key(g), shape_bits(g), panel_color(g), energy(g)
c, sh, col, e = st()
r = plan(move, fam, pickups, (c, sh, col), e, (45, 9), None, None, forbid=tuple(MECH))
for a in r[0]:
    f = step(game, a)
print("首次到(45,9):", st())
# 出去再回来, 找非机关邻居
nb = None
for a, back in ((1,2),(2,1),(3,4),(4,3)):
    n = move.get(((45,9), a))
    if n and n != (45,9) and n not in MECH:
        nb, out_a, in_a = n, a, back; break
print("邻居", nb)
for i in range(3):
    f = step(game, out_a); e1 = energy(np.array(f.frame[-1]))
    f = step(game, in_a);  e2 = energy(np.array(f.frame[-1]))
    print(f"  第{i+2}次进入: 出格后{e1} -> 进格后{e2} {'仍在补' if e2 > e1 else '不再补(一次性)'}")
