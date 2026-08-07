"""L6 通关: 两段整体规划(带能量下限, 逼它绕路吃一次性补给) + 分岔触发式重规划。"""
import json
import numpy as np
from plan_l6 import load, plan, step_rule
from wm import Percept, energy, load_env, panel_color, shape_bits, step

move, fam, pickups = load()
pickups = pickups - {(50, 24)}          # 实测: 起点格是死亡重生点, 假补给
for r in (30, 35, 40):
    move[((r, 54), 2)] = (r + 5, 54)
    move[((r + 5, 54), 1)] = (r, 54)

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


def run_leg(cell, sh_t, col_t, min_e=0, rounds=40):
    for _ in range(rounds):
        c, sh, col, e = st()
        if c == cell and sh == sh_t and col == col_t:
            return True
        r = plan(move, fam, pickups, (c, sh, col), e, cell, sh_t, col_t, min_e=min_e)
        if r is None:
            r = plan(move, fam, pickups, (c, sh, col), e, cell, sh_t, col_t)
            if r is None:
                print("  无解:", st()); return False
        for a in r[0]:
            dst = move.get((c, a), c)
            pred = step_rule(c, sh, col, dst, fam)
            if not do(a):
                print("  死亡"); return False
            c2, sh2, col2, _ = st()
            if pred is None or (c2, sh2, col2) != (dst, pred[0], pred[1]):
                break                        # 分岔 -> 重规划
            c, sh, col = c2, sh2, col2
    return st()[:3] == (cell, sh_t, col_t)


print("起始", st(), flush=True)
if not run_leg((30, 54), 413, 8, min_e=20):
    raise SystemExit(f"第一段失败: {st()}")
print("锁A前", st(), flush=True)
do(2)
print("穿锁A", st(), flush=True)
if st()[0] != (35, 54):
    raise SystemExit("锁A未开")
if not run_leg((45, 54), 485, 9):
    raise SystemExit(f"第二段失败: {st()}")
print("锁B前", st(), flush=True)
do(2)
if f.frame and f.levels_completed > base:
    print(f"*** L6 通关!!! 本关 {len(seq)} 步")
    json.dump({"level6_seq": seq}, open("l6_seq.json", "w"))
else:
    print("锁B未开:", st() if f.frame else "死亡")
