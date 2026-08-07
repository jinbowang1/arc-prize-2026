"""L6 第二段: 真模拟器搜索 + 离线模型算出的精确启发式。

上一版启发式是"格距 + 形状不对罚6 + 颜色不对罚4", 常数罚项没有梯度,
最佳优先退化成广度搜索(2万次扩展 h 在 11~24 反复横跳)。

这一版把启发式换成**离线模型上到目标的真实剩余步数**:
在离线转移图上枚举可达的 (格,形状,颜色), 建反图, 从目标反向 BFS 得到精确距离表。
它忽略能量约束, 所以是下界(可采纳); 真机只在移动机关处偏离离线模型。

另加两条剪枝:
  - 饿死剪枝: 能量不够走到目标也不够走到任一补给 => 这条分支注定饿死
  - 能量分档去重: (格,形状,颜色,能量//4), 比全精度能量去重小一个数量级
"""
import copy, heapq, json, time
from collections import deque
import numpy as np
from plan_l6 import load, plan, step_rule
from wm import Percept, energy, load_env, panel_color, shape_bits, step
from arcengine import GameAction, ActionInput

ACTS = {a: getattr(GameAction, f"ACTION{a}") for a in (1, 2, 3, 4)}
GOAL, T_SH, T_COL = (45, 54), 485, 9
W = 1.3                                   # 加权 A*: 牺牲最优性换速度
move, fam, pickups = load(); pickups = pickups - {(50, 24)}
for r in (30, 35, 40):
    move[((r, 54), 2)] = (r + 5, 54); move[((r + 5, 54), 1)] = (r, 54)


def build_dist(start):
    """离线模型上 (格,形状,颜色) -> 到目标的最小步数。"""
    edges, seen, stack = [], {start}, [start]
    while stack:
        s = stack.pop()
        cell, sh, col = s
        for a in (1, 2, 3, 4):
            dst = move.get((cell, a))
            if dst is None:
                continue
            nx = step_rule(cell, sh, col, dst, fam)
            if nx is None:
                continue
            s2 = (dst, nx[0], nx[1])
            edges.append((s, s2))
            if s2 not in seen:
                seen.add(s2); stack.append(s2)
    rev = {}
    for u, v in edges:
        rev.setdefault(v, []).append(u)
    goal = (GOAL, T_SH, T_COL)
    if goal not in seen:
        return None, seen
    dist, dq = {goal: 0}, deque([goal])
    while dq:
        u = dq.popleft()
        for v in rev.get(u, []):
            if v not in dist:
                dist[v] = dist[u] + 1; dq.append(v)
    return dist, seen


def build_dpick():
    """格 -> 到最近补给点的最小步数(忽略形状颜色)。"""
    dist = {q: 0 for q in pickups}
    dq = deque(pickups)
    rev = {}
    for (cell, a), dst in move.items():
        rev.setdefault(dst, []).append(cell)
    while dq:
        u = dq.popleft()
        for v in rev.get(u, []):
            if v not in dist:
                dist[v] = dist[u] + 1; dq.append(v)
    return dist


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


# ===== 第一段(已验证): 带 形状413+色8 走到锁A前, 穿锁 =====
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

# ===== 第二段: 精确启发式 + 真机搜索 =====
g0 = np.array(f.frame[-1])
s0 = (p.key(g0), shape_bits(g0), panel_color(g0), energy(g0))
DIST, reach = build_dist(s0[:3])
if DIST is None:
    raise SystemExit(f"离线模型里目标 {(GOAL, T_SH, T_COL)} 从 {s0[:3]} 不可达(枚举了{len(reach)}态)")
FAR = max(DIST.values()) + 8
DPICK = build_dpick()
print(f"启发表: 可达态{len(reach)} 有距离{len(DIST)} 起点h={DIST.get(s0[:3])} 最远{FAR-8}", flush=True)


def h(s):
    return DIST.get(s[:3], FAR)


def doomed(s):
    """能量既到不了目标也到不了任一补给 => 注定饿死。"""
    need = min(h(s), DPICK.get(s[0], 999))
    return s[3] < 2 * need


clean = game._clean_levels
key0 = (s0[0], s0[1], s0[2], s0[3] // 4)
pq = [(W * h(s0), 0, 0, game, [], s0)]; seen = {key0}; n = 0; t0 = time.time(); tie = 0
best_h = h(s0)
while pq:
    _, g_cost, _, bg, path, s = heapq.heappop(pq)
    for a in (1, 2, 3, 4):
        ch = clone(bg, clean); fr = raw(ch, a); n += 1
        if not fr.frame: continue
        if fr.levels_completed > base:
            full = leg1 + path + [a]      # leg1 已含穿锁A那一步, 不要再补 [2]
            print(f"*** L6 通关! 第二段{len(path)+1}步, 本关共{len(full)}步", flush=True)
            json.dump({"level6_seq": full}, open("l6_seq.json", "w")); raise SystemExit
        gg = np.array(fr.frame[-1])
        s2 = (p.key(gg), shape_bits(gg), panel_color(gg), energy(gg))
        k2 = (s2[0], s2[1], s2[2], s2[3] // 4)
        if k2 in seen or s2[3] < 2 or doomed(s2): continue
        seen.add(k2); tie += 1
        if h(s2) < best_h:
            best_h = h(s2)
            print(f"  h={best_h} 深度{g_cost+1} 能量{s2[3]} 态{s2[:3]} {time.time()-t0:.0f}s", flush=True)
        heapq.heappush(pq, (g_cost + 1 + W * h(s2), g_cost + 1, tie, ch, path + [a], s2))
    if n % 2000 == 0:
        print(f"扩展{n} 队列{len(pq)} 已见{len(seen)} 最优h={best_h} {time.time()-t0:.0f}s", flush=True)
print("队列空, 无解", flush=True)
