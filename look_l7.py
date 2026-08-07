"""看 L7 的开局地图: ASCII 渲染 + 5边框候选统计。"""
import numpy as np
from wm import Percept, energy, load_env, panel_color, shape_bits

game, f = load_env("solutions_l6.json")
g = np.array(f.frame[-1])
p = Percept(g)
print(f"L7 起始: 关={f.levels_completed} 格={p.key(g)} 形状={shape_bits(g)} "
      f"色={panel_color(g)} 能量={energy(g)}")
print("色值分布:", {int(v): int(n) for v, n in zip(*np.unique(g, return_counts=True))})

CH = ".123456789ABCDEF"
print("    " + "".join(str(c // 10 % 10) for c in range(64)))
print("    " + "".join(str(c % 10) for c in range(64)))
for r in range(64):
    print(f"{r:>3} " + "".join(CH[v] for v in g[r]))

# 5边框窗口普查(不限 7x7): 看 L7 是不是换了尺寸
print("\n色5 的连通块外接框:")
mask = (g == 5)
seen = np.zeros_like(mask)
for r in range(64):
    for c in range(64):
        if not mask[r, c] or seen[r, c]:
            continue
        stack, cells = [(r, c)], []
        seen[r, c] = True
        while stack:
            y, x = stack.pop(); cells.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < 64 and 0 <= nx < 64 and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True; stack.append((ny, nx))
        ys = [y for y, _ in cells]; xs = [x for _, x in cells]
        if len(cells) >= 8:
            print(f"  行{min(ys)}-{max(ys)} 列{min(xs)}-{max(xs)} 像素{len(cells)}")
