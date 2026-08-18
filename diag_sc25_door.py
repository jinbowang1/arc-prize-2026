"""L3: 方块卡在 (31.5,28.5), 正下方 行34-36 列27-30 是 OOOO(色13)。
点九宫格能不能把这道门打开?"""
import json
import numpy as np
from harness.env import Action, Game
from harness.run import _parse

sol = json.load(open("sc25_solutions.json"))
game, obs = Game.make("sc25")
for t in sol["seq"]:
    obs = game.act(_parse(t))
g0 = np.array(obs.grid)
DOOR = (34, 36, 27, 30)
d0 = g0[34:37, 27:31]
print(f"门区 {DOOR} 现在:\n{d0}")

R, C = [49, 54, 59], [24, 29, 34]
print("\n点九宫格各格后, 门区变了没:")
for i, r in enumerate(R):
    for j, c in enumerate(C):
        a = Action.click(C[j] + 1, R[i] + 1)
        o = game.effect(a)
        if o.dead:
            continue
        d = np.array(o.grid)[34:37, 27:31]
        if not np.array_equal(d, d0):
            print(f"  格({i},{j}) -> 门区**变了** {sorted(set(d.flatten().tolist()))}")

# 门是不是被"方块推到门口"触发的?
print("\n把方块推到门口后, 门区变了没:")
n = game.fork()
for k in [2, 2, 3, 3]:
    n.act(Action.key(k))
o = n.act(Action.key(3))
d = np.array(o.grid)[34:37, 27:31]
print(f"  推到 (31.5,28.5) 后门区: {sorted(set(d.flatten().tolist()))} "
      f"{'变了' if not np.array_equal(d, d0) else '没变'}")

# 门区到底是什么颜色/形状 —— 打印周边
print("\n门区周边 行30..44 列20..44:")
PAL = " .:-=+*#%@$&XO<>"
for r in range(30, 45):
    print(f"  {r:>3} " + "".join(PAL[v % 16] for v in g0[r, 20:45]))
