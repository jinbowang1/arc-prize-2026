"""sb26 事后侦察: 用"走两次"测每个动作的真实效果。

裸跑(results/blind_sb26.log)报 19/20 个动作无效、24 个动作全部"样本不足",
但 recon_game.py 是**单步 diff**。sc25 已经证过: perform_action 只入缓冲,
下一步才结算 —— 单步看到的是上一个动作的效果。所以先排除这个假象再下结论。

判据: fork 出克隆体, 走目标动作, 再走一次同一动作, 与开局帧比。
"""
from __future__ import annotations

import numpy as np

from harness.env import Action, Game

game, obs0 = Game.make("sb26")
base = np.array(obs0.grid)
print(f"开局 level={obs0.level}/{obs0.win_levels} 动作={list(obs0.actions)}")


def effect(a: Action, times: int = 2):
    g = game.fork()
    o = None
    for _ in range(times):
        o = g.act(a)
    d = np.argwhere(np.array(o.grid) != base)
    return o, d


def describe(name: str, a: Action):
    o1, d1 = effect(a, 1)
    o2, d2 = effect(a, 2)
    def box(d):
        if not len(d):
            return "无变化"
        return (f"{len(d)}格 行{d[:,0].min()}-{d[:,0].max()} "
                f"列{d[:,1].min()}-{d[:,1].max()}")
    tag = ""
    if o2.dead:
        tag += " [GAME_OVER]"
    if o2.level != obs0.level:
        tag += f" [过关 ->L{o2.level}]"
    print(f"{name:<22} 走1次: {box(d1):<28} 走2次: {box(d2)}{tag}")


print("\n=== 键盘动作 ===")
for i in (5, 7):
    describe(f"A{i}", Action.key(i))

# 画面上的三块结构(recon 读出来的): 顶部四色框 / 中部带框面板 / 底部四色块
TARGETS = {
    "顶框1(9)": (3, 21), "顶框2(E)": (3, 28), "顶框3(B)": (3, 35), "顶框4(F)": (3, 42),
    "中面板标记1": (29, 22), "中面板标记2": (29, 28),
    "中面板标记3": (29, 34), "中面板标记4": (29, 40),
    "中面板空白": (27, 30), "面板边框": (25, 30),
    "底块1(E)": (58, 19), "底块2(F)": (58, 27),
    "底块3(9)": (58, 35), "底块4(B)": (58, 43),
    "底块间空隙": (58, 23), "背景": (45, 10), "进度条行53": (53, 30),
}

print("\n=== 点击 (A6) ===")
for name, (y, x) in TARGETS.items():
    describe(f"A6({x},{y}) {name}", Action.click(x, y, 6))

print("\n=== 点击 (A7) ===")
for name, (y, x) in list(TARGETS.items())[:8]:
    describe(f"A7({x},{y}) {name}", Action.click(x, y, 7))
