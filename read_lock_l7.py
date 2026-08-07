"""把 L7 那块疑似锁显示屏的像素逐个打出来, 人工核对 find_locks 的读数是否可信。

背景: scan 阶段在 (49,28) 读到 413、在 (50,28) 读到 19, 两个只差一行的读数互相矛盾,
说明 7x7 检测器在这张图上会误命中。搜索目标必须先钉死。
"""
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
cur = lambda: p.key(np.array(f.frame[-1]))
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
        raise SystemExit("途中死亡")

g = np.array(f.frame[-1])
print(f"站位 {cur()} 形状 {shape_bits(g)} 色 {panel_color(g)} 能量 {energy(g)}")
hits = find_locks(g)
print(f"find_locks 命中 {len(hits)} 处: {[(r, c, col, b) for r, c, col, b in hits]}\n")

for (r0, c0, col, bits) in hits:
    print(f"=== 命中 @({r0},{c0}) 判为 色{col} 形状{bits}={show(bits)} ===")
    print("     " + " ".join(f"{c:>2}" for c in range(c0, c0 + 7)))
    for r in range(r0, r0 + 7):
        row = " ".join(f"{int(g[r, c]):>2}" for c in range(c0, c0 + 7))
        mark = "  <- core行" if r0 + 2 <= r <= r0 + 4 else ""
        print(f"{r:>4} {row}{mark}")
    core = g[r0 + 2:r0 + 5, c0 + 2:c0 + 5]
    print(f"  core 3x3 =\n{core}")
    print(f"  非5色值 {sorted(set(core.flatten().tolist()) - {5})}\n")

# 面板对照: 左下角 6x6 每格 2x2
print("=== 左下面板 (55-60, 3-8) 原始像素 ===")
print("     " + " ".join(f"{c:>2}" for c in range(3, 9)))
for r in range(55, 61):
    print(f"{r:>4} " + " ".join(f"{int(g[r, c]):>2}" for c in range(3, 9)))
print(f"shape_bits={shape_bits(g)}={show(shape_bits(g))} panel_color={panel_color(g)}")
