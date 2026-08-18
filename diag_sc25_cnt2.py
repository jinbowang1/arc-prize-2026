"""L3: (画布 x 方块) 已穷尽无解, 画面上只剩计数器在变 -> 它是不是过关条件的一部分?
计数器 32 格, 每点 1 次减 2 格。测: 把它耗到不同程度, 配合画布/方块, 会不会过关。"""
import json
import numpy as np
from harness.env import Action, Game
from harness.percept import analyze
from harness.run import _parse

sol = json.load(open("sc25_solutions.json"))
game, obs = Game.make("sc25")
for t in sol["seq"]:
    obs = game.act(_parse(t))
L0 = obs.level
g0 = np.array(obs.grid)
R, C = [49, 54, 59], [24, 29, 34]
sc = analyze(obs.grid)
NOOP = next(Action.click(c, r) for (r, c) in sc.targets
            if not game.effect(Action.click(c, r)).dead
            and np.array_equal(np.array(game.effect(Action.click(c, r)).grid), g0))
click = lambda i, j: Action.click(C[j] + 1, R[i] + 1)
cnt = lambda g: int((np.array(g)[0:8, 60:64] != 0).sum())
cv = lambda g: [[int(np.array(g)[R[i]+1, C[j]+1]) for j in range(3)] for i in range(3)]

print(f"起点: 计数器 {cnt(g0)} 格, 画布 {cv(g0)}")
print("\n反复点同一格(消耗计数器), 看计数器归零时会怎样:")
n = game.fork()
won = None
for k in range(1, 40):
    o = n.act(click(0, 0))
    if o.dead:
        print(f"  第 {k} 次点击后 **死亡/GAME_OVER**"); break
    if o.level > L0:
        won = k; break
    p = n.fork().act(NOOP)
    if not p.dead and p.level > L0:
        won = f"{k}(补步)"; break
    if k % 4 == 0 or cnt(o.grid) == 0:
        print(f"  点 {k:>2} 次: 计数器 {cnt(o.grid):>2} 格, 画布 {cv(o.grid)}, level={o.level}")
    if cnt(o.grid) == 0 and k > 20:
        break
print(f"过关: {won if won else '无'}")
