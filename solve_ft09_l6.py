"""ft09 L6: lights-out(click 翻 self+up), 蓝图定目标, 逐列自底向上求解。"""
import json
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
    g1 = np.array(fr.frame[-1])
    if np.any(g1[:63] != g0[:63]):
        cells.add((y, x))
ys = sorted({y for y, _ in cells}); xs = sorted({x for _, x in cells})
dy = 8

PATS = []
for (fy, fx) in [(y, x) for y in ys for x in xs]:
    if (fy, fx) in cells:
        continue
    blk = g0[fy - 2:fy + 4, fx - 2:fx + 4]
    if blk.shape != (6, 6) or 0 not in blk:
        continue
    bp = [[int(blk[2 * i][2 * j]) for j in range(3)] for i in range(3)]
    PATS.append(((fy, fx), bp))
print(f"格 {len(cells)} 花纹 {len(PATS)}")

BASE, CLICK = 11, 14
want = {}
for (fy, fx), bp in PATS:
    fc = bp[1][1]
    oth = BASE if fc == CLICK else CLICK
    for i in range(3):
        for j in range(3):
            if (i == 1 and j == 1) or bp[i][j] == 3:
                continue
            pos = (fy + (i - 1) * dy, fx + (j - 1) * dy)
            if pos not in cells:
                continue
            tgt = fc if bp[i][j] == 0 else oth
            if pos in want and want[pos] != tgt:
                print(f"冲突@{pos}")
            want[pos] = tgt

# 目标翻转向量(want 未覆盖的格 = 保持)
need = {p: (1 if want.get(p, BASE) != BASE else 0) for p in cells}
print(f"want {len(want)} 需翻 {sum(need.values())}")

# 逐列自底向上: click(y,x) 翻 (y,x) 和 (y-8,x)(若存在)
clicks = []
for x in xs:
    col = sorted([y for (y, xx) in cells if xx == x], reverse=True)
    flip = {y: 0 for y in col}
    for y in col:   # 自底向上
        if (need[(y, x)] + flip[y]) % 2 == 1:
            clicks.append((x, y))
            flip[y] += 1
            if (y - 8, x) in flip:
                flip[y - 8] += 1
print(f"求解点击 {len(clicks)}")

ch = clone(game)
win = False
for (x, y) in clicks:
    fr = raw(ch, 6, {"x": x, "y": y})
    if fr.levels_completed >= level:
        win = True; break
print(f"clone: {'WIN' if win else '未过'}")
if win:
    for (x, y) in clicks:
        f = env.step(GameAction.ACTION6, {"x": x, "y": y})
    print(f"真机 levels={f.levels_completed} state={f.state.name}")
    if f.levels_completed >= level:
        sols["seqs"].append(clicks)
        json.dump(sols, open("ft09_solutions.json", "w"))
        print("已存 — ft09 全通!" if f.levels_completed >= 6 else "已存")
