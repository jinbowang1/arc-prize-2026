"""L6: '蓝图=点击图'假设, '.' 位=点击格, 重叠并集/XOR 两版验证。"""
import json
from collections import Counter
import numpy as np
import arc_agi
from arcengine import GameAction, ActionInput
from solve_ft09_all import blocks, clone, raw

arc = arc_agi.Arcade()
env = arc.make("ft09")
f = env.reset()
sols = json.load(open("ft09_solutions.json"))
for seq in sols["seqs"]:
    for (x, y) in seq:
        f = env.step(GameAction.ACTION6, {"x": x, "y": y})
game = env._game
g0 = np.array(f.frame[-1])
level = f.levels_completed + 1

cands = blocks(g0)
cells = set()
for (y, x) in cands:
    ch = clone(game)
    fr = raw(ch, 6, {"x": x, "y": y})
    if np.any(np.array(fr.frame[-1])[:63] != g0[:63]):
        cells.add((y, x))
ys = sorted({y for y, _ in cells}); xs = sorted({x for _, x in cells})
PATS = []
for (fy, fx) in [(y, x) for y in ys for x in xs]:
    if (fy, fx) in cells:
        continue
    blk = g0[fy - 2:fy + 4, fx - 2:fx + 4]
    if blk.shape == (6, 6) and 0 in blk:
        PATS.append(((fy, fx), [[int(blk[2*i][2*j]) for j in range(3)] for i in range(3)]))

marks = Counter()
for (fy, fx), bp in PATS:
    for i in range(3):
        for j in range(3):
            if (i == 1 and j == 1) or bp[i][j] != 0:
                continue
            pos = (fy + (i-1)*8, fx + (j-1)*8)
            if pos in cells:
                marks[pos] += 1

for mode in ("union", "xor"):
    clicks = sorted((x, y) for (y, x), n in marks.items() if (n >= 1 if mode == "union" else n % 2 == 1))
    ch = clone(game)
    win = False
    for (x, y) in clicks:
        fr = raw(ch, 6, {"x": x, "y": y})
        if fr.levels_completed >= level:
            win = True; break
    print(f"{mode}: {len(clicks)}击 {'WIN' if win else '未过'}")
    if win:
        for (x, y) in clicks:
            f = env.step(GameAction.ACTION6, {"x": x, "y": y})
        print(f"真机 levels={f.levels_completed} state={f.state.name}")
        if f.levels_completed >= level:
            sols["seqs"].append(clicks)
            json.dump(sols, open("ft09_solutions.json", "w"))
            print("已存 — ft09 全通!" if f.levels_completed >= 6 else "已存")
        break
