"""走到能看见锁的位置, 完整渲染那一屏, 人工核实锁的读数与锁房入口。"""
import ast, json, sys
from collections import deque
import numpy as np
from wm import Percept, energy, load_env, panel_color, shape_bits, step
from validate_lock import find_locks, show

TARGET = ast.literal_eval(sys.argv[1]) if len(sys.argv) > 1 else (40, 19)
d = json.load(open("l7_model.json"))
T = {ast.literal_eval(k): v for k, v in d["transitions"].items()}
START = (15, 19)
move = {}
for (state, a), v in T.items():
    cell, dst = state[0], ast.literal_eval(v[0])[0]
    if dst == START and v[1] > 0 and cell != START:
        continue
    move[(cell, a)] = dst

game, f = load_env("solutions_l6.json")
p = Percept(np.array(f.frame[-1]))


def cur(): return p.key(np.array(f.frame[-1]))


prev, dq = {cur(): None}, deque([cur()])
while dq:
    u = dq.popleft()
    if u == TARGET:
        break
    for a in (1, 2, 3, 4):
        v = move.get((u, a))
        if v is not None and v not in prev:
            prev[v] = (u, a); dq.append(v)
path, v = [], TARGET
while prev.get(v):
    u, a = prev[v]; path.append(a); v = u
for a in reversed(path):
    f = step(game, a)
    if not f.frame:
        print("途中死亡"); raise SystemExit

g = np.array(f.frame[-1])
print(f"站位 {cur()} 形状 {shape_bits(g)} 色 {panel_color(g)} 能量 {energy(g)} 用了{len(path)}步")
print("find_locks:", [(r, c, col, b, show(b)) for r, c, col, b in find_locks(g)])
CH = ".123456789ABCDEF"
print("    " + "".join(str(c // 10 % 10) for c in range(64)))
print("    " + "".join(str(c % 10) for c in range(64)))
for r in range(64):
    print(f"{r:>3} " + "".join(CH[v] for v in g[r]))
