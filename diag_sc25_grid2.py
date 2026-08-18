"""L3 枚举(修正版): 上一版补步用了点击, 反而把画布 (0,1) 从 2 变成 14 ——
画布根本没到全 2, 那个否证不成立。

补步必须用**不改变任何维度**的动作。L3 有 18 个无效点击(28 目标 - 10 有效),
正好当补步: 它们点了全屏一格不变, 但仍然推进一拍结算。
"""
import json
import numpy as np
from harness.env import Action, Game, action_space
from harness.percept import analyze
from harness.run import _parse

sol = json.load(open("sc25_solutions.json"))
game, obs = Game.make("sc25")
for t in sol["seq"]:
    obs = game.act(_parse(t))
L0 = obs.level
g0 = np.array(obs.grid)
R, C = [49, 54, 59], [24, 29, 34]
sp = action_space(list(obs.actions))
sc = analyze(obs.grid)

# 找一个真 noop 点击当补步动作
NOOP = None
for (r, c) in sc.targets:
    a = Action.click(c, r)
    o = game.effect(a)
    if not o.dead and np.array_equal(np.array(o.grid), g0):
        NOOP = a
        break
print(f"补步用的 noop 动作: {repr(NOOP)}")
assert NOOP is not None

def canvas3(g):
    return [[int(g[r + 1, c + 1]) for c in C] for r in R]
def blk(g):
    m = ((g == 9) | (g == 10)); m[37:43, 22:31] = False; m[49:62, 24:37] = False
    c = np.argwhere(m)
    return (round(float(c[:, 0].mean()), 1), round(float(c[:, 1].mean()), 1)) if len(c) else None

# 先确认: 点中间列三格 + noop 补步 -> 画布真的是全 2 吗
n = game.fork()
for i in range(3):
    n.act(Action.click(C[1] + 1, R[i] + 1))
o = n.act(NOOP)
print(f"点中间列三格 + noop 补步 -> 画布 {canvas3(np.array(o.grid))} | level={o.level}")

hits, tried = [], 0
for dn in range(0, 6):
    for lf in range(-3, 4):
        n = game.fork(); ok = True
        for _ in range(dn):
            if n.act(Action.key(2)).dead: ok = False; break
        if ok:
            for _ in range(abs(lf)):
                if n.act(Action.key(3 if lf > 0 else 4)).dead: ok = False; break
        if not ok: continue
        for i in range(3):
            if n.act(Action.click(C[1] + 1, R[i] + 1)).dead: ok = False; break
        if not ok: continue
        o = n.act(NOOP)
        tried += 1
        if o.level > L0:
            hits.append((blk(np.array(o.grid)), dn, lf))
            print(f"🏆 过关! 方块 {blk(g0)} 下{dn} 左{lf}")
        elif tried == 1:
            print(f"  样例: 方块 {blk(np.array(o.grid))} 画布 {canvas3(np.array(o.grid))} level={o.level}")
print(f"\n枚举 {tried} 个位置 x 画布全2(补步不污染): 过关 {len(hits)}")
