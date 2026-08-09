"""打印 L5 每框的 7 种环内容, 检查例句符号是否在环里。"""
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

BOX = [(10, 16, 8, 14), (10, 16, 18, 24), (10, 16, 31, 37), (10, 16, 41, 55),
       (22, 28, 8, 21), (22, 28, 25, 31), (22, 28, 38, 44), (22, 28, 48, 54)]
names = ["对1.A", "对1.B", "对2.A", "对2.B", "对3.A", "对3.B", "对4.A", "对4.B"]

def content(g, b):
    r0, r1, c0, c1 = b
    inner = g[r0 + 1:r1, c0 + 1:c1]
    out = []
    c = 0
    while c + 4 <= inner.shape[1]:
        m = inner[:, c:c + 5]
        out.append("".join("".join("X" if v == 5 else "." for v in row) + "/" for row in m)[:-1].replace("/", "/", 99))
        c += 7
    return " | ".join(".".join([]) or ("/".join(s.split("/"))) for s in out) or "(空)"

for i, (b, nm) in enumerate(zip(BOX, names)):
    ch = clone(game)
    for _ in range(i):
        raw(ch, 4)
    fr = raw(ch, 3); fr = raw(ch, 4)
    ring = []
    g0 = np.array(fr.frame[-1])
    ring.append(content(g0, b))
    for _ in range(8):
        fr = raw(ch, 1)
        cc = content(np.array(fr.frame[-1]), b)
        if cc == ring[0]:
            break
        ring.append(cc)
    print(f"{nm} 环{len(ring)}:")
    for r in ring:
        print("    ", r)
