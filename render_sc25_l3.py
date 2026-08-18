"""渲染 L3 画面: h1(画布全2) + h2(轨道清空) 都满足了仍不过关, 第三个条件是什么?"""
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
print(f"L{obs.level+1} 开局, 用到的颜色 {sorted(set(g.flatten().tolist()))}")
print("    " + "".join(str(c % 10) for c in range(64)))
BOX = (49, 61, 24, 36)      # 画布(九宫格)
CFG = (18, 29, 31, 42)      # 构型区
for r in range(64):
    row = "".join(PAL[v % 16] for v in g[r])
    tag = ""
    if BOX[0] <= r <= BOX[1]:
        tag += " <画布"
    if CFG[0] <= r <= CFG[1]:
        tag += " <构型"
    print(f"{r:>3} {row}{tag}")
