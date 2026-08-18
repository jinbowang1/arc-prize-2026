"""L3: 四个按键各把方块往哪推? 方块与插槽在哪?"""
import json
import numpy as np
from harness.env import Action, Game, action_space
from harness.run import _parse

sol = json.load(open("sc25_solutions.json"))
game, obs = Game.make("sc25")
for t in sol["seq"]:
    obs = game.act(_parse(t))
g0 = np.array(obs.grid)
sp = action_space(list(obs.actions))
acts = [Action.key(i) for i in sp["keys"]]
game.detect_lag(acts)

# 方块 = 色 9/10 组成的块; 插槽也是 9/10 —— 用位置区分
for col in (9, 10):
    cells = np.argwhere(g0 == col)
    if len(cells):
        rows = sorted({int(r) for r, _ in cells}); cs = sorted({int(c) for _, c in cells})
        print(f"色{col}: {len(cells)} 格, 行{rows[0]}..{rows[-1]} 列{cs[0]}..{cs[-1]}")

print("\n按键效果(走两次看真效果):")
for k in sp["keys"]:
    o = game.effect(Action.key(k))
    if o.dead:
        print(f"  A{k}: 致死"); continue
    g = np.array(o.grid)
    d = np.argwhere(g != g0)
    if not len(d):
        print(f"  A{k}: 无变化"); continue
    rows = sorted({int(r) for r, _ in d}); cs = sorted({int(c) for _, c in d})
    # 方块(色9/10)移动前后的中心
    def ctr(gg):
        c = np.argwhere((gg == 9) | (gg == 10))
        c = c[c[:, 0] < 34]          # 只看管道里的那块(排除下方插槽)
        return (float(c[:, 0].mean()), float(c[:, 1].mean())) if len(c) else None
    a, b = ctr(g0), ctr(g)
    mv = f"中心 {a} -> {b}" if a and b else "方块不在管道里了"
    print(f"  A{k}: 改 {len(d)} 格 行{rows[0]}..{rows[-1]} 列{cs[0]}..{cs[-1]} | {mv}")
