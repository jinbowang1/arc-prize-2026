"""L7 全图扫描: 系统遍历可达格, 每步找锁显示框, 顺便记录形状/颜色机关格。"""
import ast, json
from collections import deque
import numpy as np
from wm import Percept, energy, load_env, panel_color, shape_bits, step
from validate_lock import find_locks, show

d = json.load(open("l7_model.json"))
T = {ast.literal_eval(k): v for k, v in d["transitions"].items()}
START = (15, 19)
move = {}
for (state, a), v in T.items():
    cell = state[0]
    dst = ast.literal_eval(v[0])[0]
    if dst == START and v[1] > 0 and cell != START:
        continue                              # 死亡出口, 不走
    move[(cell, a)] = dst

game, f = load_env("solutions_l6.json")
base = f.levels_completed
p = Percept(np.array(f.frame[-1]))
locks, mech = {}, {}
visited, seq = set(), []


def stt():
    g = np.array(f.frame[-1])
    return p.key(g), shape_bits(g), panel_color(g), energy(g), g


def scan():
    c, sh, col, e, g = stt()
    for (r, cc, lc, bits) in find_locks(g):
        if (r, cc) not in locks:
            locks[(r, cc)] = (lc, bits)
            print(f"  锁 @({r},{cc}) 色{lc} 形状{bits}={show(bits)}  [站在{c}]", flush=True)
    return c


def route(dst):
    """模型上 BFS 找到 dst 的动作串。"""
    c = stt()[0]
    if c == dst:
        return []
    prev, dq = {c: None}, deque([c])
    while dq:
        u = dq.popleft()
        for a in (1, 2, 3, 4):
            v = move.get((u, a))
            if v is None or v in prev:
                continue
            prev[v] = (u, a)
            if v == dst:
                path = []
                while v != c:
                    u2, a2 = prev[v]; path.append(a2); v = u2
                return path[::-1]
            dq.append(v)
    return None


targets = sorted({v for v in move.values()} | {k[0] for k in move},
                 key=lambda x: -(x[0] + x[1]))     # 先去最远处, 视口滚得最多
scan()
for t in targets:
    if t in visited:
        continue
    path = route(t)
    if path is None:
        continue
    ok = True
    for a in path:
        before = stt()[:3]
        f = step(game, a); seq.append(a)
        if not f.frame:
            print(f"  死亡 @{before}, 重来", flush=True)
            game, f = load_env("solutions_l6.json")
            p = Percept(np.array(f.frame[-1])); seq = []
            ok = False; break
        c2, sh2, col2, e2, _ = stt()
        if (sh2, col2) != before[1:]:
            mech[c2] = (before[1:], (sh2, col2))
            print(f"  机关 {c2}: {before[1:]} -> {(sh2, col2)}", flush=True)
        if f.levels_completed > base:
            print(f"*** 扫描途中通关! {len(seq)}步", flush=True)
            json.dump({"level7_seq": seq}, open("l7_seq.json", "w")); raise SystemExit
        visited.add(c2)
        if e2 < 8:                              # 快没电了, 重来省得饿死
            game, f = load_env("solutions_l6.json")
            p = Percept(np.array(f.frame[-1])); seq = []
            ok = False; break
    if ok:
        scan()

print(f"\n扫到 {len(visited)} 格, 锁 {len(locks)} 个, 机关格 {len(mech)} 个")
for k, v in sorted(locks.items()):
    print(f"  锁@{k} 色{v[0]} 形状{v[1]}={show(v[1])}")
for k, v in sorted(mech.items()):
    print(f"  机关@{k} {v[0]} -> {v[1]}")
