"""L5 诊断: cells 全表 + 每花纹 want 分配 + 冲突/覆盖检查。"""
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
cells = {}
for (y, x) in cands:
    ch = clone(game)
    fr = raw(ch, 6, {"x": x, "y": y})
    g1 = np.array(fr.frame[-1])
    if np.any(g1[:63] != g0[:63]):
        blk = g0[y - 2:y + 4, x - 2:x + 4]
        uniq = sorted(set(blk.flatten().tolist()))
        cells[(y, x)] = (int(g0[y, x]), uniq, int(g1[y, x]))
print("cells (y,x): (质心色, 块内色集, 点后色)")
for k, v in sorted(cells.items()):
    print("  ", k, v)

ys = sorted({y for y, _ in cells}); xs = sorted({x for _, x in cells})
dy = ys[1] - ys[0]
print(f"格网 y={ys} x={xs} dy={dy}")
want = {}
for (fy, fx) in [(y, x) for y in ys for x in xs]:
    if (fy, fx) in cells:
        continue
    blk = g0[fy - 2:fy + 4, fx - 2:fx + 4]
    if blk.shape != (6, 6) or 0 not in blk:
        continue
    bp = [[int(blk[2 * i][2 * j]) for j in range(3)] for i in range(3)]
    fc = bp[1][1]
    oth = 15 if fc == 14 else 14
    print(f"花纹@({fy},{fx}) 中心{fc} bp={bp}")
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                continue
            pos = (fy + (i - 1) * dy, fx + (j - 1) * dy)
            v = bp[i][j]
            if v == 3:
                tag = "越界" if pos not in cells else f"⚠️3但有格{cells[pos][0]}"
                if pos in cells:
                    print(f"    {pos}: {tag}")
                continue
            if pos not in cells:
                print(f"    {pos}: ⚠️蓝图{v}但无格")
                continue
            tgt = fc if v == 0 else oth
            if pos in want and want[pos] != tgt:
                print(f"    {pos}: 🚨冲突 已有{want[pos]} 新要求{tgt}")
            want[pos] = tgt
uncov = [k for k in cells if k not in want]
print(f"未覆盖格: {uncov}")
need = [(k, cells[k][0], v) for k, v in sorted(want.items()) if cells[k][0] != v]
print(f"需改 {len(need)}: {need}")
