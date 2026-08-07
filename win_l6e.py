"""L6 收官: 穿上锁(永久开启) -> 回西边配 485+色9 -> 穿走廊开下锁。"""
import json
import numpy as np
from plan_l6 import load, plan, step_rule, ROTS, XFORMS
from wm import Percept, energy, load_env, panel_color, shape_bits, step

move, fam, pickups = load()
# 补进实测的走廊连通(探索没到过锁后区域)
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


def reach(cell, sh=None, col=None, forbid=(), rounds=14):
    for _ in range(rounds):
        c, s, k, e = st()
        if c == cell and (sh is None or s == sh) and (col is None or k == col):
            return True
        r = plan(move, fam, pickups, (c, s, k), e, cell, sh, col, forbid=forbid)
        if r is None:
            return False
        for a in r[0]:
            dst = move.get((c, a), c)
            pred = step_rule(c, s, k, dst, fam)
            if not do(a):
                return False
            c2, s2, k2, _ = st()
            if pred is None or (c2, s2, k2) != (dst, pred[0], pred[1]):
                break
            c, s, k = c2, s2, k2
    return False


print("起始", st())
if not reach((40, 19)):
    raise SystemExit("到不了旋转带")
for i in range(14):
    if st()[1] == 413:
        break
    do(3 if i % 2 == 0 else 4)
if st()[1] != 413:
    raise SystemExit(f"没转出413: {st()}")
if not reach((30, 54), 413, 8, forbid=tuple(ROTS | XFORMS)):
    raise SystemExit(f"到不了上锁前: {st()}")
do(2)                                        # 穿上锁(此后永久开启)
print("已穿上锁", st())

# 回西边: 先进换族带拿到 359 族, 再去旋转带转到 485, 最后配色9
FAM359 = {359, 485, 461, 335}
if st()[1] not in FAM359:
    if not reach((10, 24)):
        raise SystemExit(f"到不了换族带: {st()}")
    for i in range(14):
        if st()[1] in FAM359: break
        do(3 if i % 2 == 0 else 4)
print("换族后", st())
if st()[1] not in FAM359:
    raise SystemExit("没进359族")
if not reach((40, 24), forbid=tuple(XFORMS)):
    raise SystemExit(f"到不了旋转带: {st()}")
for i in range(14):
    if st()[1] == 485: break
    do(3 if i % 2 == 0 else 4)
print("形状485就位", st())
if st()[1] != 485:
    raise SystemExit(f"没转出485: {st()}")
PAL = {(r,c) for r in (20,25,30) for c in (19,24,29)}
F_ALL = tuple(ROTS | XFORMS | PAL)
do(2); do(2)                           # (40,24)->(45,24)->(50,24) 补给点在两格下
print("补给后", st())
if not reach((25, 34), 485, None, forbid=F_ALL):
    raise SystemExit(f"到不了调色区东缘: {st()}")
print("东缘就位", st())
for i in range(10):                    # 在 (25,34)<->(25,29) 来回, 每次进入推进一格颜色
    if st()[2] == 9: break
    do(3); do(4)
print("色9就位", st())
if st()[2] != 9:
    raise SystemExit(f"没调出9: {st()}")
if not reach((45, 54), 485, 9, forbid=F_ALL):
    raise SystemExit(f"到不了下锁前: {st()}")
print("下锁前", st())
do(2)
if f.frame and f.levels_completed > base:
    print(f"*** L6 通关! 本关 {len(seq)} 步")
    json.dump({"level6_seq": seq}, open("l6_seq.json", "w"))
else:
    print("下锁未开:", st() if f.frame else "死亡")
