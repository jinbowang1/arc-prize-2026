"""L1 过关前那一帧的全屏 —— 找我还没注意到的第三个变量。
之前只看了画布(九宫格)和轨道, 现在逐格看全屏。"""
import json
import numpy as np
from harness.env import Game
from harness.run import _parse

PAL = " .:-=+*#%@$&XO<>"
sol = json.load(open("sc25_solutions.json"))
seq = [_parse(t) for t in sol["seq"]]
n1 = sol["per_level_steps"][0]

game, obs = Game.make("sc25")
g_start = np.array(obs.grid)
for a in seq[:n1 - 1]:                 # 走到过关前一步
    obs = game.act(a)
g_before = np.array(obs.grid)
print(f"L1 过关前(走了 {n1-1} 步), level={obs.level}")
print("    " + "".join(str(c % 10) for c in range(64)))
for r in range(64):
    line = "".join(PAL[v % 16] for v in g_before[r])
    mark = "  <>" if not np.array_equal(g_before[r], g_start[r]) else ""
    print(f"{r:>3} {line}{mark}")

d = np.argwhere(g_before != g_start)
rows = sorted({int(r) for r, _ in d})
print(f"\n相对开局改了 {len(d)} 格, 涉及行: {rows}")
