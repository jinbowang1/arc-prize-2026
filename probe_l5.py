"""L5 定向探针 v2: 模式匹配定位钥匙块(上2行C+下3行9)。"""
import json
import numpy as np
import arc_agi
from arcengine import GameAction

A = {1: GameAction.ACTION1, 2: GameAction.ACTION2, 3: GameAction.ACTION3, 4: GameAction.ACTION4}
PREFIX = json.load(open("solutions.json"))["seq"]


def key_pos(g):
    for r in range(0, 60):
        for c in range(13, 60):
            if r + 5 > 64 or c + 5 > 64:
                continue
            win = g[r:r + 5, c:c + 5]
            if (win[0:2] == 12).all() and (win[2:5] == 9).all():
                return (r, c)
    return None


def panel(g):
    out = []
    for br in range(3):
        row = ""
        for bc in range(3):
            blk = g[55 + br * 2: 57 + br * 2, 3 + bc * 2: 5 + bc * 2]
            vals = set(blk.flatten().tolist()) - {5}
            row += "." if not vals else format(sorted(vals)[0], "X")
        out.append(row)
    return "/".join(out)


def bar(g):
    return int((g[61] == 11).sum())


def fresh():
    arc = arc_agi.Arcade()
    env = arc.make("ls20")
    env.reset()
    for a in PREFIX:
        f = env.step(A[a])
    return env, f


def run(tag, seq):
    env, f = fresh()
    g = np.array(f.frame[-1])
    print(f"== {tag} == key={key_pos(g)} bar={bar(g)} panel={panel(g)}")
    prev = key_pos(g)
    for i, a in enumerate(seq):
        f = env.step(A[a])
        g = np.array(f.frame[-1])
        kp = key_pos(g)
        blocked = " BLOCKED" if kp == prev else ""
        print(f"  s{i+1} a{a} key={kp} bar={bar(g)} panel={panel(g)} lv={f.levels_completed} {f.state.name}{blocked}")
        prev = kp


import sys
seqs = json.loads(sys.argv[1])
for tag, seq in seqs:
    run(tag, seq)
