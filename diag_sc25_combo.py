"""L3: 单个动作都不改门区(已验)。那**组合**呢?

系统枚举, 不猜: 把方块推到若干个不同位置, 每个位置上把 14 个动作**各试一遍**,
看门区(行34-36 列27-30, 色13)有没有任何反应。
顺带看 level 有没有意外上升 —— 判据永远只认它。
"""
import json
import numpy as np
from harness.env import Action, Game, action_space
from harness.percept import analyze
from harness.run import _parse

DOOR = (34, 37, 27, 31)
sol = json.load(open("sc25_solutions.json"))
game, obs = Game.make("sc25")
for t in sol["seq"]:
    obs = game.act(_parse(t))
g0 = np.array(obs.grid)
d0 = g0[DOOR[0]:DOOR[1], DOOR[2]:DOOR[3]]
sp = action_space(list(obs.actions))
sc = analyze(obs.grid)
clicks = [Action.click(c, r) for (r, c) in sc.targets]
acts = [Action.key(k) for k in sp["keys"]] + clicks
game.detect_lag(acts)

def blk(g):
    m = ((g == 9) | (g == 10)); m[37:43, 22:31] = False; m[49:62, 24:37] = False
    c = np.argwhere(m)
    return (round(float(c[:, 0].mean()), 1), round(float(c[:, 1].mean()), 1)) if len(c) else None

# 方块推到各处(用按键序列造现场)
sites = {}
for name, ks in {"原位": [], "下2": [2, 2], "下2左3": [2, 2, 3, 3, 3],
                 "左3": [3, 3, 3], "下2左3下2": [2, 2, 3, 3, 3, 2, 2],
                 "上2": [1, 1], "右2": [4, 4]}.items():
    n = game.fork(); ok = True
    for k in ks:
        if n.act(Action.key(k)).dead:
            ok = False; break
    if ok:
        sites[name] = n

print(f"门区基准:\n{d0}\n")
print(f"{'现场':<12} {'方块中心':<16} 有反应的动作")
found = []
for name, node in sites.items():
    cur = np.array(node._grid())
    hits = []
    for a in acts:
        ch = node.fork()
        o = ch.act(a)
        if o.dead:
            continue
        if o.level > obs.level:
            found.append((name, repr(a), "过关"))
            hits.append(f"{repr(a)}🏆")
            continue
        # 补一步结算(过关信号可能滞后)
        p = ch.fork().act(a)
        if not p.dead and p.level > obs.level:
            found.append((name, repr(a), "过关(补步)"))
            hits.append(f"{repr(a)}🏆补")
            continue
        d = np.array(p.grid)[DOOR[0]:DOOR[1], DOOR[2]:DOOR[3]]
        if not np.array_equal(d, d0):
            hits.append(repr(a))
    print(f"{name:<12} {str(blk(cur)):<16} {hits if hits else '无'}")
print(f"\n意外过关: {found if found else '无'}")
