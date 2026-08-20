"""把 sb26 的关键几步渲染成字符图, 看图说话。

只画有内容的行(24-35 面板 / 53 条 / 56-61 底部 / 0-7 顶部), 省得刷屏。
"""
from __future__ import annotations

import numpy as np

from harness.env import Action, Game

PAL = " .:-=+*#%@$&XO<>"
BANDS = [(0, 7, "顶部"), (24, 35, "面板"), (52, 54, "条"), (56, 61, "底部")]


def draw(grid, tag):
    g = np.array(grid)
    print(f"\n----- {tag} -----")
    for lo, hi, name in BANDS:
        print(f"  [{name}]")
        for r in range(lo, hi + 1):
            row = "".join(PAL[c] if c < len(PAL) else "?" for c in g[r][14:50])
            print(f"  {r:>3} {row}")


game, obs0 = Game.make("sb26")
draw(obs0.grid, "开局")

g = game.fork()
o = g.act(Action.click(19, 58, 6))
draw(o.grid, "① 点底块1(E) —— 选中")

o = g.act(Action.click(22, 29, 6))
draw(o.grid, "② 再点面板标记1 —— 施加")

o = g.act(Action.key(7))
draw(o.grid, "③ 按 A7 —— 是撤销吗?")

o = g.act(Action.key(5))
draw(o.grid, "④ 按 A5")
print(f"\nlevel={o.level} state={o.state} 条剩={int((np.array(o.grid)[53]==2).sum())}")
