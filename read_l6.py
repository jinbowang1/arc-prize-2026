"""L6 全量数据: 12 框环内容 + 题面 3 符号 + 答案 6 符号。"""
import copy, json
import numpy as np
import arc_agi
from arcengine import GameAction, ActionInput
from parse_tr87 import show

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

def glyphs(g, r0, c0, c1):
    out = []
    c = c0
    while c + 4 <= c1:
        out.append(tuple(map(tuple, [[1 if g[r0 + i][c + j] == 5 else 0 for j in range(5)] for i in range(5)])))
        c += 7
    return out

exA = glyphs(g0, 45, 22, 42)
exB = glyphs(g0, 54, 12, 53)
print("题面 A(3):")
for m in exA: print("  ", show(np.array(m)))
print("答案 B(6):")
for m in exB: print("  ", show(np.array(m)))

BOX = []
for band, r in enumerate((5, 17, 29)):
    BOX += [(r, 10, 14), (r, 20, 32), (r, 39, 43), (r, 49, 53)]
names = [f"带{t+1}.{x}" for t in range(3) for x in ("A", "77", "7", "B")]

rings = []
for i, (r, c0, c1) in enumerate(BOX):
    ch = clone(game)
    for _ in range(i):
        raw(ch, 4)
    fr = raw(ch, 3); fr = raw(ch, 4)
    ring = [glyphs(np.array(fr.frame[-1]), r + 1, c0, c1)]
    for _ in range(8):
        fr = raw(ch, 1)
        cc = glyphs(np.array(fr.frame[-1]), r + 1, c0, c1)
        if cc == ring[0]:
            break
        ring.append(cc)
    rings.append(ring)
    print(f"{names[i]} 环{len(ring)}:")
    for m in ring:
        print("    " + " | ".join(show(np.array(x)) for x in m))

import pickle
pickle.dump({"exA": exA, "exB": exB, "rings": rings, "BOX": BOX}, open("l6_data.pkl", "wb"))
print("已存 l6_data.pkl")
