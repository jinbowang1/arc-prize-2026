"""L6 上锁: 分岔触发式重规划(CEGIS 纪律), 不每步重来也不随机破局。"""
import json
import numpy as np
from plan_l6 import load, plan, step_rule, ROTS, XFORMS
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


def reach(goal_cell, goal_sh=None, goal_col=None, forbid=(), rounds=12):
    """按计划走, 只在预测与实测分岔时才重规划。"""
    for rd in range(rounds):
        c, sh, col, e = st()
        if c == goal_cell and (goal_sh is None or sh == goal_sh) and (goal_col is None or col == goal_col):
            return True
        r = plan(move, fam, pickups, (c, sh, col), e, goal_cell, goal_sh, goal_col, forbid=forbid)
        if r is None:
            print(f"  第{rd}轮无路径, 现状 {st()}")
            return False
        for a in r[0]:
            dst = move.get((c, a), c)
            pred = step_rule(c, sh, col, dst, fam)
            if not do(a):
                return False
            c2, sh2, col2, _ = st()
            if pred is None or (c2, sh2, col2) != (dst, pred[0], pred[1]):
                print(f"  分岔@{c}act{a}: 预测{(dst, pred[0] if pred else '?', pred[1] if pred else '?')} 实测{(c2, sh2, col2)} -> 重规划")
                break
            c, sh, col = c2, sh2, col2
    return False


print("起始", st())
if not reach((40, 19)):
    raise SystemExit("到不了旋转带")
for i in range(14):
    if st()[1] == 413:
        break
    do(3 if i % 2 == 0 else 4)
print("形状就位", st())
if st()[1] != 413:
    raise SystemExit("没转出413")

if not reach((30, 54), 413, 8, forbid=tuple(ROTS | XFORMS)):
    raise SystemExit(f"到不了锁前+色8: {st()}")
print("锁前", st())
do(2)
if f.frame and f.levels_completed > base:
    print(f"*** L6 通关! 本关 {len(seq)} 步")
    json.dump({"level6_seq": seq}, open("l6_seq.json", "w"))
else:
    print("进入锁格后继续向下探路:")
    for i in range(10):
        c0 = st()[0]
        if not do(2): print("  死亡"); break
        c1, sh, col, e = st()
        print(f"  下{i+1}: {c0} -> {c1} 形状{sh} 色{col} 能量{e} lv={f.levels_completed}")
        if f.levels_completed > base: print("  *** 通关!"); break
        if c1 == c0:
            print("  到底, 测能否原路返回(锁A是否单向):")
            for j in range(4):
                do(1)
            from validate_lock import find_locks
            g = np.array(f.frame[-1])
            print("    穿锁后两屏读数:", [(r,c,col,b) for r,c,col,b in find_locks(g)])
            print("    当前:", st())
            break
