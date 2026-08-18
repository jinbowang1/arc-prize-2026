"""L3: 中列初始 0, 点击是 0<->14, 永远变不成 2 -> "画布全2"物理不可达。
那目标只能是 中列=14 边列=2(= 显示器的竖线图案)。精确测: 怎么点能到这个态?
补步一律用真 noop, 不污染。"""
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
cv = lambda g: [[int(np.array(g)[R[i]+1, C[j]+1]) for j in range(3)] for i in range(3)]
click = lambda i, j: Action.click(C[j] + 1, R[i] + 1)

print(f"起点画布 {cv(g0)} | noop={repr(NOOP)}")
print("\n中列三格各点 N 次(每次序列末尾补 noop 结算), 画布变化:")
for n in range(1, 5):
    nd = game.fork()
    ok = True
    for _ in range(n):
        for i in range(3):
            if nd.act(click(i, 1)).dead:
                ok = False; break
        if not ok: break
    if not ok: continue
    o = nd.act(NOOP)
    won = o.level > L0
    p = nd.fork().act(NOOP)
    won2 = (not p.dead) and p.level > L0
    print(f"  各点 {n} 次 -> 画布 {cv(o.grid)} level={o.level}"
          f"{'  🏆过关' if won or won2 else ''}")

print("\n目标态 [[2,14,2],[2,14,2],[2,14,2]] 能不能到? 逐格试:")
nd = game.fork()
for i in range(3):
    nd.act(click(i, 1))
o = nd.act(NOOP)
print(f"  中列各点 1 次 + noop -> {cv(o.grid)}")
