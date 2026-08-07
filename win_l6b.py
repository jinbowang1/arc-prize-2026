"""L6 上锁: 沿用 L5 验证过的混合执行 —— 导航靠模型, 机关格靠真机贪心。"""
import json
import numpy as np
from plan_l6 import load, plan, ROTS, XFORMS, PALETTE
from wm import Percept, energy, load_env, panel_color, shape_bits, step

move, fam, pickups = load()
game, f = load_env("solutions_l5.json")
base = f.levels_completed
p = Percept(np.array(f.frame[-1]))
seq = []


def st():
    g = np.array(f.frame[-1])
    return p.key(g), shape_bits(g), panel_color(g), energy(g)


def do(a):
    global f
    f = step(game, a); seq.append(a)
    return bool(f.frame)


def nav(to, forbid=()):
    c, sh, col, e = st()
    if c == to:
        return True
    r = plan(move, fam, pickups, (c, sh, col), e, to, None, None, forbid=forbid)
    if r is None:
        return False
    return all(do(a) for a in r[0])


def wiggle(want, idx, limit=14):
    for i in range(limit):
        if st()[idx] == want:
            return True
        if not do(3 if i % 2 == 0 else 4):
            return False
    return st()[idx] == want


print("起始", st())
if not nav((40, 19)):
    print("到不了旋转带"); raise SystemExit
if not wiggle(413, 1):
    print("没转出413:", st()); raise SystemExit
print("形状就位", st())
# 一次性规划到锁前, 要求到达时颜色=8(沿途调色格由规划器自行计数), 禁行改形状的格
c, sh, col, e = st()
F = tuple(ROTS | XFORMS)
r = plan(move, fam, pickups, (c, sh, col), e, (30, 54), None, 8, forbid=F)
if r is None:
    print("无路径: 到锁前且色为8"); raise SystemExit
print(f"末段规划 {len(r[0])} 步")
for a in r[0]:
    if not do(a):
        print("中途死亡"); raise SystemExit
print("锁前", st())
do(2)
if f.frame and f.levels_completed > base:
    print(f"*** L6 通关! 本关 {len(seq)} 步")
    json.dump({"level6_seq": seq}, open("l6_seq.json", "w"))
else:
    print("上锁未开:", st() if f.frame else "死亡")
