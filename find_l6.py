"""放宽的锁查找: 边框任意单色即可, 并 dump L6 地图。"""
import numpy as np
from wm import load_env
from check_color import panel
CH = ".123456789ABCDEF"
game, f = load_env("solutions_l5.json")
g = np.array(f.frame[-1])
print("L6 面板:", panel(g))
found = []
for r in range(0, 57):
    for c in range(0, 57):
        w = g[r:r+7, c:c+7]
        border = np.concatenate([w[0], w[6], w[:, 0], w[:, 6]])
        if len(set(border.tolist())) != 1:
            continue
        core = w[2:5, 2:5]
        vals = set(core.flatten().tolist()) - {int(border[0])}
        if len(vals) != 1:
            continue
        col = vals.pop()
        bits = sum(1 << (i*3+j) for i in range(3) for j in range(3) if core[i, j] == col)
        found.append((r, c, int(border[0]), col, bits))
print("放宽后找到的显示屏:", found)
print("=== L6 地图 ===")
for r in range(0, 53):
    print(f"{r:2d}", "".join(CH[v] for v in g[r]))
