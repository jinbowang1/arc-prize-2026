"""ft09 L1: 块状态环确认 + 组合穷举 + 真机执行。"""
import copy, itertools, json, time
import numpy as np
import arc_agi
from arcengine import GameAction, ActionInput

def raw(g, a, data=None):
    return g.perform_action(ActionInput(id=getattr(GameAction, f"ACTION{a}"), data=data or {}), raw=True)

def clone(g):
    clean = getattr(g, "_clean_levels", None)
    g._clean_levels = None; g2 = copy.deepcopy(g)
    g._clean_levels = clean; g2._clean_levels = clean
    return g2

arc = arc_agi.Arcade()
env = arc.make("ft09")
f = env.reset()
game = env._game
g0 = np.array(f.frame[-1])

CELLS = [(38, 40), (38, 48), (38, 56), (46, 40), (46, 56), (54, 40), (54, 48), (54, 56)]

def cell_sig(g, y, x):
    return tuple(map(tuple, g[y - 2:y + 4, x - 2:x + 4]))

# 1. 单块状态环
ch = clone(game)
y, x = CELLS[0]
sig0 = cell_sig(g0, y, x)
ring = [sig0]
for i in range(5):
    fr = raw(ch, 6, {"x": x, "y": y})
    s = cell_sig(np.array(fr.frame[-1]), y, x)
    if s == sig0:
        break
    ring.append(s)
print(f"块(38,40) 状态环长: {len(ring)}")

R = len(ring)
# 2. 穷举: 每块 0..R-1 次点击
t0 = time.time()
found = None
tried = 0
for combo in itertools.product(range(R), repeat=8):
    if sum(combo) == 0:
        continue
    ch = clone(game)
    win = False
    for (yy, xx), k in zip(CELLS, combo):
        for _ in range(k):
            fr = raw(ch, 6, {"x": xx, "y": yy})
            if fr.levels_completed > 0:
                win = True; break
        if win:
            break
    tried += 1
    if win:
        found = combo
        print(f"命中! combo={combo} ({tried}试, {time.time()-t0:.0f}s)")
        break
print(f"共试 {tried} ({time.time()-t0:.0f}s)")

if found:
    seq = []
    for (yy, xx), k in zip(CELLS, found):
        seq += [(xx, yy)] * k
    for (xx, yy) in seq:
        f = env.step(GameAction.ACTION6, {"x": xx, "y": yy}) if False else env.step_action(None)
    print("真机待执行")
