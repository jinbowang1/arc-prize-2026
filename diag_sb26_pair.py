"""sb26 机制第二问: 单击是"选中"开关, 那两次点不同块会发生什么?

diag_sb26.py 已定: 点底块一次改 20 格(该块周围出现高亮框), 再点同一块
恢复原样 —— 典型的选中/取消开关。猜测: 选中 A 再点 B = 交换两块。
这里只验猜测, 不预设交换的是颜色还是整块。
"""
from __future__ import annotations

import numpy as np

from harness.env import Action, Game

game, obs0 = Game.make("sb26")
base = np.array(obs0.grid)

BLOCKS = {1: (58, 19), 2: (58, 27), 3: (58, 35), 4: (58, 43)}
ROW = 58            # 底部四块所在行
COLS = {1: 19, 2: 27, 3: 35, 4: 43}


def bottom(grid) -> list[int]:
    g = np.array(grid)
    return [int(g[ROW, c]) for c in COLS.values()]


def top() -> list[int]:
    return [int(base[3, c]) for c in (21, 28, 35, 42)]


print(f"顶部四框颜色   = {top()}")
print(f"底部四块颜色   = {bottom(obs0.grid)}")
print(f"进度条行53 剩余 = {int((base[53] == 2).sum())} 格\n")

for i in (1, 2, 3, 4):
    for j in (1, 2, 3, 4):
        if i >= j:
            continue
        g = game.fork()
        yi, xi = BLOCKS[i][0], BLOCKS[i][1]
        yj, xj = BLOCKS[j][0], BLOCKS[j][1]
        g.act(Action.click(xi, yi, 6))
        o = g.act(Action.click(xj, yj, 6))
        after = bottom(o.grid)
        d = np.argwhere(np.array(o.grid) != base)
        bar = int((np.array(o.grid)[53] == 2).sum())
        note = "交换成功" if after != bottom(obs0.grid) else "底部未变"
        print(f"选{i}->点{j}: 底部={after} 改动{len(d)}格 条={bar} "
              f"level={o.level} {note}")
