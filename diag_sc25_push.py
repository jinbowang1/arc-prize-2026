"""L3: 手工把方块往插槽推, 看卡在哪。A1上 A2下 A3左 A4右, 每次 4 格。"""
import json
import numpy as np
from harness.env import Action, Game
from harness.run import _parse

sol = json.load(open("sc25_solutions.json"))
game, obs = Game.make("sc25")
for t in sol["seq"]:
    obs = game.act(_parse(t))

def ctr(g):
    m = ((g == 9) | (g == 10))
    m[37:43, 22:31] = False          # 排除插槽本身
    m[49:62, 24:37] = False          # 排除画布
    c = np.argwhere(m)
    return (round(float(c[:, 0].mean()), 1), round(float(c[:, 1].mean()), 1)) if len(c) else None

for label, seq in {
    "下x4 再 左x3": [2]*4 + [3]*3,
    "左x3 再 下x4": [3]*3 + [2]*4,
    "下x2 左x3 下x2": [2]*2 + [3]*3 + [2]*2,
    "只下x6":        [2]*6,
    "只左x5":        [3]*5,
}.items():
    n = game.fork(); o = None; path = []
    for k in seq:
        o = n.act(Action.key(k))
        if o.dead: path.append("DEAD"); break
        if o.level > obs.level: path.append(f"🏆过关"); break
        path.append(str(ctr(np.array(o.grid))))
    print(f"{label:<16} -> {' '.join(path)}")
