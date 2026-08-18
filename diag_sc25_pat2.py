"""L3: 画布全2 已被否证(42 位置全不过关)。显示器图案是**中间列高亮**,
试"中间列点成 14"等几种图案 x 方块位置。补步用真 noop, 不污染。"""
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
NOOP = None
for (r, c) in sc.targets:
    a = Action.click(c, r)
    o = game.effect(a)
    if not o.dead and np.array_equal(np.array(o.grid), g0):
        NOOP = a; break
cv = lambda g: [[int(g[r+1, c+1]) for c in C] for r in R]
click = lambda i, j: Action.click(C[j] + 1, R[i] + 1)

# 几种候选图案(用"每格点几次"表示; 颜色环 0->2->14)
PATTERNS = {
    "中列点2次(->14)":      {(i,1): 2 for i in range(3)},
    "中列点1次+边列点1次":   {**{(i,1):1 for i in range(3)}, **{(i,j):1 for i in range(3) for j in (0,2)}},
    "边列点1次(->14)":      {(i,j):1 for i in range(3) for j in (0,2)},
    "全格点1次":            {(i,j):1 for i in range(3) for j in range(3)},
    "中列点3次":            {(i,1):3 for i in range(3)},
}
print(f"noop 补步动作 {repr(NOOP)}; 开局画布 {cv(g0)}\n")
for pname, spec in PATTERNS.items():
    won = []
    for dn in range(0, 6):
        for lf in range(-3, 4):
            n = game.fork(); ok = True
            for _ in range(dn):
                if n.act(Action.key(2)).dead: ok = False; break
            if ok:
                for _ in range(abs(lf)):
                    if n.act(Action.key(3 if lf > 0 else 4)).dead: ok = False; break
            if not ok: continue
            for (i, j), cnt in spec.items():
                for _ in range(cnt):
                    if n.act(click(i, j)).dead: ok = False; break
                if not ok: break
            if not ok: continue
            o = n.act(NOOP)
            if o.level > L0:
                won.append((dn, lf))
    # 该图案实际长什么样(方块不动的情形)
    n = game.fork()
    for (i, j), cnt in spec.items():
        for _ in range(cnt):
            n.act(click(i, j))
    o = n.act(NOOP)
    print(f"{pname:<22} 画布 {cv(np.array(o.grid))} | 过关位置 {won if won else '无'}")
