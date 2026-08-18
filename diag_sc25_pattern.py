"""L3: 画布目标真是"全色2"吗? 还是要匹配显示器的图案(不同像素密度的同一图案)?

L1/L2 过关瞬间画布都是全色 2 —— 但那可能只是**那两关的图案恰好是全满**。
ls20 早有先例: 锁显示是 7x7 框正中 3x3 每格 1 像素, 面板是 6x6 每格 2x2 像素,
**同一个图案、不同像素密度**。
"""
import json
import numpy as np
from harness.env import Game
from harness.run import _parse

PAL = " .:-=+*#%@$&XO<>"
sol = json.load(open("sc25_solutions.json"))
game, obs = Game.make("sc25")
for t in sol["seq"]:
    obs = game.act(_parse(t))
g = np.array(obs.grid)

print("显示器区 行46-62 列8-26:")
for r in range(46, 63):
    print(f"  {r:>3} " + "".join(PAL[v % 16] for v in g[r, 8:27]))

print("\n画布(九宫格)区 行47-63 列22-38:")
for r in range(47, 64):
    print(f"  {r:>3} " + "".join(PAL[v % 16] for v in g[r, 22:39]))

# 画布 3x3 采样(每格中心)
R, C = [49, 54, 59], [24, 29, 34]
print(f"\n画布 3x3 各格中心色: {[[int(g[r+1, c+1]) for c in C] for r in R]}")
