"""L5: 光标遍历(ACTION4)与当前框符号环(ACTION1)。"""
import copy, json
import numpy as np
import arc_agi
from arcengine import GameAction, ActionInput

ACTS = {i: getattr(GameAction, f"ACTION{i}") for i in range(1, 8)}
def raw(g, a): return g.perform_action(ActionInput(id=ACTS[a], data={}), raw=True)
def clone(g):
    clean = getattr(g, "_clean_levels", None)
    g._clean_levels = None; g2 = copy.deepcopy(g)
    g._clean_levels = clean; g2._clean_levels = clean
    return g2

arc = arc_agi.Arcade()
env = arc.make("tr87")
f = env.reset()
sols = json.load(open("tr87_solutions.json"))
for seq in sols["seqs"]:
    for a in seq:
        f = env.step(ACTS[a])
game = env._game
g0 = np.array(f.frame[-1])

def cursor(g):
    d = np.argwhere(g == 0)
    if len(d) == 0:
        return None
    rs, cs = d[:, 0], d[:, 1]
    return (int(rs.min()), int(rs.max()), int(cs.min()), int(cs.max()))

print(f"初始光标: {cursor(g0)}")
ch = clone(game)
for i in range(10):
    fr = raw(ch, 4)
    print(f"  ACTION4 x{i+1}: 光标 {cursor(np.array(fr.frame[-1]))} 关卡={fr.levels_completed}")

ch = clone(game)
print("\nACTION1 在初始框循环:")
prev = g0
for i in range(10):
    fr = raw(ch, 1)
    g = np.array(fr.frame[-1])
    d = np.argwhere(g != prev)
    rows = sorted({int(r) for r, _ in d})
    cols = sorted({int(c) for _, c in d})
    print(f"  x{i+1}: 变动{len(d)}px 行{rows[:3]}..{rows[-1:] if rows else ''} 列{cols[:2]}..{cols[-1:] if cols else ''} 关卡={fr.levels_completed}")
    prev = g
