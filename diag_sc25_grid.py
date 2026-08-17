"""sc25: 九宫格是不是可切换的答案区, 显示器是不是题面?

看图得到的结构(render_sc25.log):
    行19-22 列23-42  轨道 + 一个 2x4 双色块(@@/$$), A1/A2 旋转 A3 左移 A4 翻转
    行50-58 列11-20  图案显示器(色15 的 >> 组成图案)
    行47-63 列22-38  3x3 九宫格(每格 3x3 像素), 点击效果恒定 9 格
    九宫格现在 X.X/.X./X.X, 显示器色15 分布 .X./X.X/.X. —— 正好互为反色

这一轮只问四件事:
    ① 点九宫格某格, 是切换那一格, 还是别的
    ② 九宫格的 3x3 布尔图 与 显示器的 3x3 布尔图 到底怎么对应
    ③ 有没有一串点击能让 level 上升(直接试"把九宫格点成显示器的样子")
    ④ 双色块(A1-A4)在这里起什么作用 —— 会不会也影响九宫格
"""
from __future__ import annotations

import numpy as np

from harness.env import Action, Game, action_space
from harness.percept import analyze

game, obs = Game.make("sc25")
g0 = np.array(obs.grid)
sp = action_space(list(obs.actions))
sc = analyze(obs.grid)
acts = [Action.key(i) for i in sp["keys"]] + [Action.click(c, r) for (r, c) in sc.targets]
game.detect_lag(acts)

# 九宫格: 行 49/54/59 起, 列 24/29/34 起, 每格 3x3
CELL_R = [49, 54, 59]
CELL_C = [24, 29, 34]
# 显示器: 8x8 内区在 行51..58 列12..19, 按 2 行 x 2 列一个单元看不齐,
# 先把色15 的位置原样打出来再判断
def grid3(g):
    return np.array([[1 if g[r, c] != 0 else 0 for c in CELL_C] for r in CELL_R])

print("[九宫格 3x3 布尔图(非空=1)]", flush=True)
print(grid3(g0), flush=True)

print("\n[显示器区域 行50..59 列11..20 逐行]", flush=True)
for r in range(50, 60):
    print(f"  行{r}: " + "".join("X" if g0[r, c] == 15 else ("." if g0[r, c] == 2 else " ")
                                 for c in range(11, 21)), flush=True)

print("\n[① 点九宫格每格的效果]", flush=True)
for i, r in enumerate(CELL_R):
    for j, c in enumerate(CELL_C):
        a = Action.click(c + 1, r + 1)
        o = game.effect(a)
        if o.dead:
            print(f"  格({i},{j}) 点({r+1},{c+1}) 致死", flush=True); continue
        g = np.array(o.grid)
        d = np.argwhere(g != g0)
        before, after = grid3(g0), grid3(g)
        flip = np.argwhere(before != after)
        print(f"  格({i},{j}) 点({r+1},{c+1}): 改 {len(d)} 格, 3x3 图变化位置 {flip.tolist()}",
              flush=True)

print("\n[③ 把九宫格全点一遍(9 次点击), level 会不会动]", flush=True)
n = game.fork()
o = None
for i, r in enumerate(CELL_R):
    for j, c in enumerate(CELL_C):
        o = n.act(Action.click(c + 1, r + 1))
        if o.dead:
            break
o2 = n.act(Action.click(CELL_C[0] + 1, CELL_R[0] + 1))   # 补一步结算最后那次点击
print(f"  9 次点击后: level={o2.level} dead={o2.dead}", flush=True)
print(f"  九宫格现在:\n{grid3(np.array(o2.grid))}", flush=True)

print("\n[④ A1-A4 会不会影响九宫格]", flush=True)
for k in sp["keys"]:
    o = game.effect(Action.key(k))
    if o.dead:
        continue
    g = np.array(o.grid)
    same = np.array_equal(grid3(g), grid3(g0))
    print(f"  A{k}: 九宫格 3x3 图不变? {same}", flush=True)
