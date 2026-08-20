"""sb26 机制第四问: 底部四块是不是调色板 —— 选色之后去中间面板"画"?

已定:
    点底块 = 排他性选中(20 格高亮), 底部本身从不改变
    A5     = 消耗第 53 行的条 + 清掉选中;  A7 = 无作用
    空手点面板 = 无变化
所以剩下的唯一组合是"先选色, 再点面板"。面板是 rows 24-35 那个 8 号边框框,
里面有四对 2 号标记(行29-30, 列22/28/34/40 附近)。
"""
from __future__ import annotations

import numpy as np

from harness.env import Action, Game

game, obs0 = Game.make("sb26")
base = np.array(obs0.grid)
PAL = " .:-=+*#%@$&XO<>"

BLOCKS = {1: (58, 19), 2: (58, 27), 3: (58, 35), 4: (58, 43)}


def diff(o):
    g = np.array(o.grid)
    return np.argwhere(g != base), g


def show(tag, o):
    d, g = diff(o)
    if not len(d):
        print(f"{tag:<34} 无变化")
        return
    print(f"{tag:<34} 改{len(d):>3}格 行{d[:,0].min()}-{d[:,0].max()} "
          f"列{d[:,1].min()}-{d[:,1].max()} 条={int((g[53]==2).sum())} level={o.level}")


print("=== 选色 -> 点面板内不同位置 ===")
SPOTS = {"标记1(29,22)": (29, 22), "标记2(29,28)": (29, 28),
         "标记3(29,34)": (29, 34), "标记4(29,40)": (29, 40),
         "面板空白(27,30)": (27, 30), "面板中心(30,31)": (30, 31)}
for bi in (1, 2):
    for name, (y, x) in SPOTS.items():
        g = game.fork()
        by, bx = BLOCKS[bi]
        g.act(Action.click(bx, by, 6))
        o = g.act(Action.click(x, y, 6))
        show(f"选块{bi} -> 点{name}", o)
    print()

print("=== 选色 -> 点顶部空心框内部 ===")
for bi in (1, 2):
    for name, (y, x) in {"顶框1内(3,20)": (3, 20), "顶框2内(3,27)": (3, 27),
                         "顶框3内(3,34)": (3, 34), "顶框4内(3,41)": (3, 41)}.items():
        g = game.fork()
        by, bx = BLOCKS[bi]
        g.act(Action.click(bx, by, 6))
        o = g.act(Action.click(x, y, 6))
        show(f"选块{bi} -> 点{name}", o)
    print()

print("=== 选色 -> A7 (确认?) -> 看面板 ===")
for bi in (1, 3):
    g = game.fork()
    by, bx = BLOCKS[bi]
    g.act(Action.click(bx, by, 6))
    g.act(Action.click(29, 22, 6))
    o = g.act(Action.key(7))
    show(f"选块{bi} -> 点标记1 -> A7", o)
