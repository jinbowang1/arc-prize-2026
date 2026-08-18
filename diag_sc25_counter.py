"""L3: 穷尽 (画布 x 方块位置) 无解, 而画面上只剩计数器在变。
计数器是不是第三维度? 便宜实验: 造两个 (画布,方块) 相同、计数器不同的状态,
看它们行为是否不同。

造法: 同一格点两次(切换两次回原样) + noop 补步 -> 画布/方块都复原, 计数器却走了。
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
sc = analyze(obs.grid)
sp = action_space(list(obs.actions))
acts = [Action.key(k) for k in sp["keys"]] + [Action.click(c, r) for (r, c) in sc.targets]
game.detect_lag(acts)
NOOP = next(Action.click(c, r) for (r, c) in sc.targets
            if not game.effect(Action.click(c, r)).dead
            and np.array_equal(np.array(game.effect(Action.click(c, r)).grid), g0))

CNT = (0, 8, 60, 64)          # 右上计数器区
cv = lambda g: [[int(g[R[i]+1, C[j]+1]) for j in range(3)] for i in range(3)]
def blk(g):
    sub = g[10:34, 26:46]; m = ((sub == 9) | (sub == 10)); c = np.argwhere(m)
    return (round(float(c[:,0].mean())+10,1), round(float(c[:,1].mean())+26,1)) if len(c) else None
cnt = lambda g: g[CNT[0]:CNT[1], CNT[2]:CNT[3]].copy()

c00 = Action.click(C[0] + 1, R[0] + 1)
print(f"noop 动作 {repr(NOOP)}")
print(f"起点: 画布 {cv(g0)} 方块 {blk(g0)} 计数器非0格 {int((cnt(g0)!=0).sum())}\n")

# 造 N 组"画布/方块相同、计数器不同"的状态
states = [("原状态", game.fork(), g0)]
n = game.fork()
for k in range(1, 5):
    n.act(c00); n.act(c00)          # 切换两次 -> 画布复原
    o = n.act(NOOP)                 # 补步结算
    g = np.array(o.grid)
    states.append((f"多点 {2*k} 次", n.fork(), g))

print(f"{'状态':<12} {'画布相同?':<10} {'方块相同?':<10} {'计数器非0格':<12}")
for name, nd, g in states:
    print(f"{name:<12} {str(cv(g)==cv(g0)):<10} {str(blk(g)==blk(g0)):<10} {int((cnt(g)!=0).sum()):<12}")

# 关键: 从这些状态出发走同一批动作, 行为(尤其 level)是否不同
print("\n从各状态走同一批动作, 看是否有过关差异:")
probe = [Action.key(2), Action.key(3), c00, Action.key(2), NOOP]
for name, nd, g in states:
    m = nd.fork(); lv = None; dead = False
    for a in probe:
        o = m.act(a)
        if o.dead: dead = True; break
        lv = o.level
    print(f"  {name:<12} 走完 {len(probe)} 步 -> level={lv} dead={dead}")
