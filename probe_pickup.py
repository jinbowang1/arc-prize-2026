"""补给机制体检: 四个候选点逐个实测——真的假的?回满还是定量?一次性还是可重复?"""
import numpy as np
from plan_l6 import load, plan, step_rule, ROTS, XFORMS, PALETTE
from wm import Percept, energy, load_env, panel_color, shape_bits, step

MECH = ROTS | XFORMS | PALETTE
move, fam, pickups = load()
print("候选补给点:", sorted(pickups))
for target in sorted(pickups):
    game, f = load_env("solutions_l5.json")
    p = Percept(np.array(f.frame[-1]))
    def st():
        g = np.array(f.frame[-1]); return p.key(g), shape_bits(g), panel_color(g), energy(g)
    c, sh, col, e = st()
    if c == target:
        print(f"  {target}: 就是起点格, 无法测入格效果(高度可疑=死亡重生点)"); continue
    r = plan(move, fam, pickups, (c, sh, col), e, target, None, None, forbid=tuple(MECH))
    if r is None:
        print(f"  {target}: 规划不到"); continue
    prev = e
    for a in r[0]:
        before = energy(np.array(f.frame[-1]))
        f = step(game, a)
        if not f.frame: break
        after = energy(np.array(f.frame[-1]))
        if after > before:
            print(f"  {target}: 第{r[0].index(a)+1}步 能量 {before} -> {after} (+{after-before}) 位置{p.key(np.array(f.frame[-1]))}")
    c2, _, _, e2 = st()
    print(f"  {target}: 到达{c2} 能量{e2} (出发{e}, 走了{len(r[0])}步应耗{2*len(r[0])})"
          f" => {'有补给' if e2 > e - 2*len(r[0]) else '无补给'}")
