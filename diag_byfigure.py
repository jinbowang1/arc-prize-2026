"""by_figure 为什么把 5x5 的钥匙吞进跨屏巨块? 先看它认的背景色对不对。"""
import json
import numpy as np
from harness.env import Action, Game
from harness.percept import background, by_figure

seq = json.load(open("solutions.json"))["seq"]
game, obs = Game.make("ls20")
g = np.array(obs.grid)
print("开局各色格数(前 6):")
vals, cnts = np.unique(g, return_counts=True)
for v, c in sorted(zip(vals.tolist(), cnts.tolist()), key=lambda x: -x[1])[:6]:
    print(f"   色 {v:>2}: {c:>5} 格 ({c/g.size:.0%})")
bg = background(g)
print(f"\nbackground() 认的背景色 = {bg}")

blobs = by_figure(g, bg)
print(f"by_figure 分出 {len(blobs)} 块:")
for b in blobs[:8]:
    r0, r1, c0, c1 = b.bbox
    print(f"   bbox={b.bbox} 尺寸 {r1-r0+1}x{c1-c0+1} 格数 {len(b.cells) if hasattr(b,'cells') else '?'}")

# 钥匙在哪? 找 5x5 的小块
print("\n若改用**按颜色分别连通**(每种颜色单独找连通块), 会分出什么:")
from collections import deque
for col in sorted(set(g.flatten().tolist())):
    if col == bg:
        continue
    m = (g == col)
    seen = np.zeros_like(m)
    parts = []
    for r in range(64):
        for c in range(64):
            if not m[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)]); seen[r, c] = True; cells = []
            while q:
                y, x = q.popleft(); cells.append((y, x))
                for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                    yy, xx = y+dy, x+dx
                    if 0 <= yy < 64 and 0 <= xx < 64 and m[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True; q.append((yy, xx))
            parts.append(cells)
    small = [p for p in parts if 4 <= len(p) <= 200]
    if small:
        print(f"   色 {col:>2}: {len(parts)} 块, 其中 4~200 格的有 {len(small)} 个 "
              f"{[(min(y for y,_ in p), min(x for _,x in p), len(p)) for p in small[:4]]}")
