"""L6 上锁 闭环版: 每步重规划, 模型出错当场纠正(而非开环跑完整条路)。"""
import json
import numpy as np
from plan_l6 import load, plan, ROTS, XFORMS
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


def closed_loop(goal_cell, goal_sh, goal_col, forbid=(), limit=60):
    """每步重规划: 对模型误差免疫。"""
    tried = {}
    for _ in range(limit):
        c, sh, col, e = st()
        if c == goal_cell and (goal_sh is None or sh == goal_sh) and (goal_col is None or col == goal_col):
            return True
        r = plan(move, fam, pickups, (c, sh, col), e, goal_cell, goal_sh, goal_col, forbid=forbid)
        if r is None or not r[0]:
            return False
        a = r[0][0]
        k = (c, sh, col, a)
        tried[k] = tried.get(k, 0) + 1
        if tried[k] > 2:                       # 卡死: 换个动作破局
            a = next((x for x in (1, 2, 3, 4) if (c, sh, col, x) not in tried), a)
        if not do(a):
            return False
    return False


print("起始", st())
if not closed_loop((40, 19), None, None):
    print("到不了旋转带"); raise SystemExit
for i in range(14):                            # 旋转带真机贪心
    if st()[1] == 413:
        break
    do(3 if i % 2 == 0 else 4)
print("形状就位", st())
if st()[1] != 413:
    print("没转出413"); raise SystemExit

if not closed_loop((30, 54), 413, 8, forbid=tuple(ROTS | XFORMS)):
    print("闭环也到不了(锁前+色8):", st()); raise SystemExit
print("锁前", st())
do(2)
if f.frame and f.levels_completed > base:
    print(f"*** L6 通关! 本关 {len(seq)} 步")
    json.dump({"level6_seq": seq}, open("l6_seq.json", "w"))
else:
    print("上锁未开:", st() if f.frame else "死亡")
