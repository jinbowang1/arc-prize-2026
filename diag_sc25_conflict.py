"""两个实验对同一操作给出冲突结果, 查清楚是谁错了。

  diag_sc25_color: 格(0,0)连点 -> [14,2,14,2,14,2]  (在切换)
  diag_sc25_cnt2 : 格(0,0)连点 -> 画布一直 [[2,0,2]] (不动)

同一坐标、同一起点, 结果不该不同。逐步打印两种读法。
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
a00 = Action.click(C[0] + 1, R[0] + 1)
print(f"起点 level={obs.level}, 点击动作 = {repr(a00)}")
print(f"格(0,0) 中心像素 = grid[{R[0]+1}][{C[0]+1}]\n")

n = game.fork()
print(f"{'第几次点':>8} {'中心像素':>8} {'该格 3x3 全部像素':>20}")
for k in range(1, 9):
    o = n.act(a00)
    g = np.array(o.grid)
    center = int(g[R[0] + 1, C[0] + 1])
    block = g[R[0]:R[0]+3, C[0]:C[0]+3]
    print(f"{k:>8} {center:>8}   {sorted(set(block.flatten().tolist()))}")
