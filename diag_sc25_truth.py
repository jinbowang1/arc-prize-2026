"""干净重测 L3 画布的转移规则。不预设模型, 每格独立测, 直接读像素。

⚠️上一轮"画布纹丝不动"是我**自己脚本的取值写错**造成的假象, 不是游戏行为。
"""
import json
import numpy as np
from harness.env import Action, Game
from harness.run import _parse

sol = json.load(open("sc25_solutions.json"))
game, obs = Game.make("sc25")
for t in sol["seq"]:
    obs = game.act(_parse(t))
R, C = [49, 54, 59], [24, 29, 34]
g0 = np.array(obs.grid)
px = lambda g, i, j: int(np.array(g)[R[i] + 1, C[j] + 1])

print("九宫格初始色:")
print("  ", [[px(g0, i, j) for j in range(3)] for i in range(3)])
print("\n逐格连点 6 次(直接读该格中心像素):")
for i in range(3):
    for j in range(3):
        n = game.fork()
        vals = []
        for _ in range(6):
            o = n.act(Action.click(C[j] + 1, R[i] + 1))
            if o.dead:
                vals.append("D"); break
            vals.append(px(o.grid, i, j))
        print(f"  格({i},{j}) 初始 {px(g0,i,j):>2} -> 连点: {vals}")
