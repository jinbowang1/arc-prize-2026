"""L6 第二段: 真模拟器 + 启发式最佳优先搜索(比 BFS 快得多)。

启发: 到锁B前的曼哈顿格距 + 形状/颜色未就位的惩罚。
所有转移仍在真模拟器上实走 => 找到的解按构造即已验证。
"""
import copy, heapq, json, time
import numpy as np
from plan_l6 import load, plan, step_rule
from wm import Percept, energy, load_env, panel_color, shape_bits, step
from arcengine import GameAction, ActionInput

ACTS = {a: getattr(GameAction, f"ACTION{a}") for a in (1, 2, 3, 4)}
GOAL, T_SH, T_COL = (45, 54), 485, 9
move, fam, pickups = load(); pickups = pickups - {(50, 24)}
for r in (30, 35, 40):
    move[((r, 54), 2)] = (r + 5, 54); move[((r + 5, 54), 1)] = (r, 54)

def raw(g, a): return g.perform_action(ActionInput(id=ACTS[a], data={}), raw=True)
def clone(g, clean):
    g._clean_levels = None; g2 = copy.deepcopy(g); g._clean_levels = clean
    g2._clean_levels = clean; return g2

game, f = load_env("solutions_l5.json")
base = f.levels_completed
p = Percept(np.array(f.frame[-1])); seq = []
def stt():
    g = np.array(f.frame[-1]); return p.key(g), shape_bits(g), panel_color(g), energy(g)
def do(a):
    global f
    f = step(game, a); seq.append(a); return bool(f.frame)

# 第一段(已验证)
for _ in range(30):
    c, sh, col, e = stt()
    if (c, sh, col) == ((30, 54), 413, 8): break
    r = plan(move, fam, pickups, (c, sh, col), e, (30, 54), 413, 8, min_e=20) \
        or plan(move, fam, pickups, (c, sh, col), e, (30, 54), 413, 8)
    if r is None: raise SystemExit(f"第一段无解 {stt()}")
    for a in r[0]:
        dst = move.get((c, a), c); pred = step_rule(c, sh, col, dst, fam)
        if not do(a): raise SystemExit("第一段死亡")
        c2, sh2, col2, _ = stt()
        if pred is None or (c2, sh2, col2) != (dst, pred[0], pred[1]): break
        c, sh, col = c2, sh2, col2
print("锁A前", stt(), flush=True)
do(2); print("穿锁A", stt(), flush=True)
leg1 = list(seq)

def h(s):
    (r, c), sh, col, e = s
    d = (abs(r - GOAL[0]) + abs(c - GOAL[1])) // 5
    return d + (6 if sh != T_SH else 0) + (4 if col != T_COL else 0)

clean = game._clean_levels
g0 = np.array(f.frame[-1])
s0 = (p.key(g0), shape_bits(g0), panel_color(g0), energy(g0))
pq = [(h(s0), 0, 0, game, [], s0)]; seen = {s0}; n = 0; t0 = time.time(); tie = 0
while pq:
    _, g_cost, _, bg, path, s = heapq.heappop(pq)
    for a in (1, 2, 3, 4):
        ch = clone(bg, clean); fr = raw(ch, a); n += 1
        if not fr.frame: continue
        if fr.levels_completed > base:
            full = leg1 + [2] + path + [a]
            print(f"*** L6 通关! 第二段{len(path)+1}步, 本关共{len(full)}步", flush=True)
            json.dump({"level6_seq": full}, open("l6_seq.json", "w")); raise SystemExit
        gg = np.array(fr.frame[-1])
        s2 = (p.key(gg), shape_bits(gg), panel_color(gg), energy(gg))
        if s2 in seen or s2[3] < 1: continue
        seen.add(s2); tie += 1
        heapq.heappush(pq, (g_cost + 1 + h(s2), g_cost + 1, tie, ch, path + [a], s2))
    if n % 2000 == 0:
        print(f"扩展{n} 队列{len(pq)} 已见{len(seen)} 最优h={h(s)} {time.time()-t0:.0f}s", flush=True)
print("队列空, 无解", flush=True)
