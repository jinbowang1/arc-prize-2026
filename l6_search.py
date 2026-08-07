"""L6 第二段: 真模拟器语义态搜索(cell, shape, color, energy)。

不信模型对移动机关的预测 —— 所有转移都在真模拟器上实走, 找到的路按构造即已验证。
第一段沿用已验证的整体规划(带能量下限), 到锁A后切搜索。
"""
import copy
import json
import time
import numpy as np
from plan_l6 import load, plan, step_rule
from wm import Percept, energy, load_env, panel_color, shape_bits, step
from arcengine import GameAction, ActionInput

ACTS = {a: getattr(GameAction, f"ACTION{a}") for a in (1, 2, 3, 4)}
move, fam, pickups = load()
pickups = pickups - {(50, 24)}
for r in (30, 35, 40):
    move[((r, 54), 2)] = (r + 5, 54)
    move[((r + 5, 54), 1)] = (r, 54)


def raw(game, a):
    return game.perform_action(ActionInput(id=ACTS[a], data={}), raw=True)


def clone(game, clean):
    game._clean_levels = None
    g2 = copy.deepcopy(game)
    game._clean_levels = clean
    g2._clean_levels = clean
    return g2


# ---- 第一段: 已验证的规划执行 ----
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


for _ in range(30):
    c, sh, col, e = st()
    if (c, sh, col) == ((30, 54), 413, 8):
        break
    r = plan(move, fam, pickups, (c, sh, col), e, (30, 54), 413, 8, min_e=20) \
        or plan(move, fam, pickups, (c, sh, col), e, (30, 54), 413, 8)
    if r is None:
        raise SystemExit(f"第一段无解 {st()}")
    for a in r[0]:
        dst = move.get((c, a), c)
        pred = step_rule(c, sh, col, dst, fam)
        if not do(a):
            raise SystemExit("第一段死亡")
        c2, sh2, col2, _ = st()
        if pred is None or (c2, sh2, col2) != (dst, pred[0], pred[1]):
            break
        c, sh, col = c2, sh2, col2
print("锁A前", st(), f"({len(seq)}步)", flush=True)
do(2)
print("穿锁A", st(), flush=True)
leg1 = list(seq)

# ---- 第二段: 真模拟器 BFS ----
clean = game._clean_levels
g0 = np.array(f.frame[-1])
s0 = (p.key(g0), shape_bits(g0), panel_color(g0), energy(g0))
frontier = [(game, [], s0)]
seen = {s0}
t0 = time.time()
for depth in range(90):
    nxt = []
    for bg, path, s in frontier:
        for a in (1, 2, 3, 4):
            ch = clone(bg, clean)
            fr = raw(ch, a)
            if not fr.frame:
                continue
            if fr.levels_completed > base:
                full = leg1 + [2] + path + [a]
                print(f"*** L6 通关! 第二段{len(path)+1}步, 本关共{len(full)}步", flush=True)
                json.dump({"level6_seq": full}, open("l6_seq.json", "w"))
                raise SystemExit
            gg = np.array(fr.frame[-1])
            s2 = (p.key(gg), shape_bits(gg), panel_color(gg), energy(gg))
            if s2 in seen or s2[3] < 1:
                continue
            seen.add(s2)
            nxt.append((ch, path + [a], s2))
    frontier = nxt
    print(f"depth {depth+1}: frontier {len(frontier)}, seen {len(seen)}, {time.time()-t0:.0f}s", flush=True)
    if not frontier:
        print("搜索穷尽, 第二段在此抽象下无解", flush=True)
        break
