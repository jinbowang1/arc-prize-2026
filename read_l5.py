"""读 L5 字典 8 框与例句, 找对应关系。"""
import json
import numpy as np
import arc_agi
from arcengine import GameAction
from parse_tr87 import show

ACTS = {i: getattr(GameAction, f"ACTION{i}") for i in range(1, 8)}
arc = arc_agi.Arcade()
env = arc.make("tr87")
f = env.reset()
sols = json.load(open("tr87_solutions.json"))
for seq in sols["seqs"]:
    for a in seq:
        f = env.step(ACTS[a])
g = np.array(f.frame[-1])

def glyphs(g, r0, c0, c1):
    out = []
    c = c0
    while c + 4 <= c1:
        m = tuple(tuple(1 if g[r0 + i][c + j] == 5 else 0 for j in range(5)) for i in range(5))
        out.append(m)
        c += 7
    return out

BOX = [(11, 9, 13), (11, 19, 23), (11, 32, 36), (11, 42, 53),
       (23, 9, 20), (23, 26, 30), (23, 39, 43), (23, 49, 53)]
names = ["对1.A", "对1.B", "对2.A", "对2.B", "对3.A", "对3.B", "对4.A", "对4.B"]
print("=== 字典 8 框 ===")
for (r, c0, c1), nm in zip(BOX, names):
    gs = glyphs(g, r, c0, c1)
    print(f"  {nm}: " + " | ".join(show(np.array(m)) for m in gs))
print("=== 例句 A 串(行44-48) ===")
for m in glyphs(g, 44, 15, 47):
    print("  ", show(np.array(m)))
print("=== 例句 7 串(行53-57) ===")
for m in glyphs(g, 53, 15, 48):
    print("  ", show(np.array(m)))
