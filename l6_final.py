"""L6 完整解(重写版): 每一步都读实测状态, 机关格自动找非机关邻居来回摇。

已知机制: 锁A(35,54)=形状413+色8, 过后永久开启; 锁B(50,54)=形状485+色9
  形状: row40=旋转带(转90°CW) / row10=换族带
  颜色: rows20,25,30 × cols19,24,29, 环 14->8->12->9->14
  补给: (5,9) (5,39) (45,9) (50,24), 回满42
  走廊: (30,54)-(35,54)-(40,54)-(45,54) 上下相连
"""
import json
import numpy as np
from plan_l6 import load, plan, step_rule, ROTS, XFORMS, PALETTE
from wm import Percept, energy, load_env, panel_color, shape_bits, step

MECH = ROTS | XFORMS | PALETTE
move, fam, pickups = load()
pickups = pickups - {(50, 24)}   # 实测: 起点格=死亡重生点, 是假补给(学习器误记)
for r in (30, 35, 40):                      # 补进实测走廊
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


def goto(cell, forbid=(), rounds=10):
    """分岔触发式重规划; 只求到位, 不约束形状颜色。"""
    for _ in range(rounds):
        c, sh, col, e = st()
        if c == cell:
            return True
        r = plan(move, fam, pickups, (c, sh, col), e, cell, None, None, forbid=forbid)
        if r is None:
            return False
        for a in r[0]:
            dst = move.get((c, a), c)
            pred = step_rule(c, sh, col, dst, fam)
            if not do(a):
                return False
            c2, sh2, col2, _ = st()
            if pred is None or (c2, sh2, col2) != (dst, pred[0], pred[1]):
                break                        # 分岔 -> 重规划
            c, sh, col = c2, sh2, col2
    return st()[0] == cell


def neighbor_of(mech_cell):
    """找该机关格的一个非机关邻居, 返回 (邻居, 去邻居的动作, 回机关的动作)。"""
    for a, back in ((1, 2), (2, 1), (3, 4), (4, 3)):
        n = move.get((mech_cell, a))
        if n and n != mech_cell and n not in MECH:
            return n, a, back
    return None, None, None


def tune(mech_cell, idx, want, limit=10):
    """站到机关格旁, 反复进入直到目标维达成。idx: 1=形状 2=颜色"""
    nb, out_a, in_a = neighbor_of(mech_cell)
    if nb is None:
        print(f"  {mech_cell} 找不到非机关邻居"); return False
    if not goto(nb, forbid=tuple(MECH - {mech_cell})):
        print(f"  到不了 {mech_cell} 的邻居 {nb}"); return False
    for _ in range(limit):
        if st()[idx] == want:
            return True
        if st()[3] < 26:
            refill(min_energy=42)
            if not goto(nb, forbid=tuple(MECH - {mech_cell})):
                return st()[idx] == want
        if not do(in_a):
            return False
        if st()[idx] == want:
            return True
        if not do(out_a):
            return False
    return st()[idx] == want


def tune_until(mech_cell, pred, limit=8):
    """进出机关格直到 pred(状态) 为真(比 tune 更通用: 用谓词而非固定目标值)。"""
    nb, out_a, in_a = neighbor_of(mech_cell)
    if nb is None or not goto(nb, forbid=tuple(MECH - {mech_cell})):
        return False
    for _ in range(limit):
        if pred(st()):
            return True
        if st()[3] < 26:                      # 能量见底: 先补给再回来接着摇
            refill(min_energy=42)
            if not goto(nb, forbid=tuple(MECH - {mech_cell})):
                return pred(st())
        if not do(in_a):
            return False
        if pred(st()):
            return True
        if not do(out_a):
            return False
    return pred(st())


def refill(min_energy=30):
    """按规划器算的真实代价挑补给点; 走不到就不去(免得半路饿死丢形状颜色)。"""
    c, sh, col, e = st()
    if e >= min_energy:
        return True
    cands = []
    for q in pickups:
        r = plan(move, fam, pickups, (c, sh, col), e, q, None, None, forbid=tuple(MECH))
        if r is not None:
            cands.append((len(r[0]), q))
    if not cands:
        print(f"  无可达补给点(能量{e})")
        return False
    cands.sort()
    print(f"  去补给 {cands[0][1]} (需{cands[0][0]}步, 现有能量{e})")
    return goto(cands[0][1], forbid=tuple(MECH))


def phase(name):
    print(f"[{name}] {st()}", flush=True)


phase("起始")
# ===== 锁A: 形状413 + 色8 =====
if not tune((40, 24), 1, 413):
    raise SystemExit(f"配不出413: {st()}")
phase("形状413")
if not tune((25, 29), 2, 8):
    raise SystemExit(f"配不出色8: {st()}")
phase("色8")
if not refill(min_energy=42):
    print("  补给失败, 继续")
phase("补满后进锁A")
if not goto((30, 54), forbid=tuple(MECH)):
    raise SystemExit(f"到不了锁A前: {st()}")
phase("锁A前")
do(2)
phase("穿锁A")
if st()[0] != (35, 54):
    raise SystemExit("锁A未开")

# ===== 锁B: 形状485 + 色9 =====
do(1)                                        # 退回 (30,54)
FAM = {359, 485, 461, 335}


def full():
    """补到满: 逐个试补给点, 用规划器算真实代价。"""
    c, sh, col, e = st()
    if e >= 40:
        return True
    cands = []
    for q in pickups:
        r = plan(move, fam, pickups, (c, sh, col), e, q, None, None, forbid=tuple(MECH))
        if r is not None and len(r[0]) * 2 < e:      # 走得到才去
            cands.append((len(r[0]), q))
    if not cands:
        return False
    cands.sort()
    return goto(cands[0][1], forbid=tuple(MECH))


def safe_tune(mech, ok, sessions=6, per=3):
    """每轮先补满再摇 per 次, 绝不摇到饿死。"""
    nb, out_a, in_a = neighbor_of(mech)
    for _ in range(sessions):
        if ok(st()):
            return True
        if not goto(nb, forbid=tuple(MECH - {mech})):   # 先到机关旁
            return ok(st())
        if st()[3] < 30:                                 # 再就近补给, 然后回来
            full()
            if not goto(nb, forbid=tuple(MECH - {mech})):
                return ok(st())
        for _ in range(per):
            if ok(st()):
                return True
            if st()[3] < 14:
                break
            do(in_a)
            if ok(st()):
                return True
            do(out_a)
    return ok(st())


if not safe_tune((10, 24), lambda s: s[1] in FAM):
    raise SystemExit(f"没进359族: {st()}")
phase("换族")
if not safe_tune((40, 24), lambda s: s[1] == 485):
    raise SystemExit(f"配不出485: {st()}")
phase("形状485")
full()
phase("补满")
# 末段: 一次性规划(沿途调色格由规划器计数), 分岔即重规划
for _ in range(8):
    c, sh, col, e = st()
    if c == (45, 54) and sh == 485 and col == 9:
        break
    r = plan(move, fam, pickups, (c, sh, col), e, (45, 54), 485, 9,
             forbid=tuple(ROTS | XFORMS))
    if r is None:
        raise SystemExit(f"末段无解: {st()}")
    for a in r[0]:
        dst = move.get((c, a), c)
        pred = step_rule(c, sh, col, dst, fam)
        do(a)
        c2, sh2, col2, _ = st()
        if pred is None or (c2, sh2, col2) != (dst, pred[0], pred[1]):
            break
        c, sh, col = c2, sh2, col2
phase("锁B前")
do(2)
if f.frame and f.levels_completed > base:
    print(f"*** L6 通关! 本关 {len(seq)} 步")
    json.dump({"level6_seq": seq}, open("l6_seq.json", "w"))
else:
    print("锁B未开:", st() if f.frame else "死亡")
