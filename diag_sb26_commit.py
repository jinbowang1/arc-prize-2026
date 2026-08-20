"""sb26 机制第三问: 选中之后, A5 / A7 对被选中的块做什么?

已定(diag_sb26.py / diag_sb26_pair.py):
    点底块 = 排他性"选中", 只改 20 格高亮框, 点第二块只是把高亮挪过去, 不交换
    A5     = 消耗第 53 行那条 64 格的条(每按一次少一格), 点击不消耗它
    A7     = 空按无变化
所以 A5/A7 很可能是"对选中目标施加操作", 空按看不出来。
"""
from __future__ import annotations

import numpy as np

from harness.env import Action, Game

game, obs0 = Game.make("sb26")
base = np.array(obs0.grid)

BLOCKS = {1: (58, 19), 2: (58, 27), 3: (58, 35), 4: (58, 43)}
COLS = {1: 19, 2: 27, 3: 35, 4: 43}
TOPCOLS = {1: 20, 2: 27, 3: 34, 4: 41}     # 顶部四个空心框的边框列


def bottom(grid):
    g = np.array(grid)
    return [int(g[58, c]) for c in COLS.values()]


def bar(grid):
    return int((np.array(grid)[53] == 2).sum())


def show(tag, o, seq_len):
    g = np.array(o.grid)
    d = np.argwhere(g != base)
    rows = f"行{d[:,0].min()}-{d[:,0].max()}" if len(d) else "-"
    print(f"{tag:<28} 底部={bottom(o.grid)} 改{len(d):>3}格 {rows:<12} "
          f"条={bar(o.grid)} level={o.level} {'GAME_OVER' if o.dead else ''}")


print(f"顶部四框边框色 = {[int(base[1, c]) for c in TOPCOLS.values()]}")
print(f"底部四块颜色   = {bottom(obs0.grid)}   条={bar(obs0.grid)}\n")

print("=== 选中一块后按 A5 ===")
for i in (1, 2, 3, 4):
    g = game.fork()
    y, x = BLOCKS[i]
    g.act(Action.click(x, y, 6))
    o = g.act(Action.key(5))
    show(f"点块{i} -> A5", o, 2)
    o = g.act(Action.key(5))
    show(f"点块{i} -> A5 A5", o, 3)

print("\n=== 选中一块后按 A7 ===")
for i in (1, 2, 3, 4):
    g = game.fork()
    y, x = BLOCKS[i]
    g.act(Action.click(x, y, 6))
    o = g.act(Action.key(7))
    show(f"点块{i} -> A7", o, 2)
    o = g.act(Action.key(7))
    show(f"点块{i} -> A7 A7", o, 3)

print("\n=== 光按 A5 若干次(不选任何东西) ===")
g = game.fork()
for n in range(1, 9):
    o = g.act(Action.key(5))
    if n in (1, 2, 4, 8):
        show(f"A5 x{n}", o, n)
