"""L6: 枚举花纹语义正/反 x 参考块参与与否, lights-out 求解 + clone 验证。"""
import itertools, json
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

BASE, CLICK = 11, 14
REF = (8, 6)

def attempt(flips, skip_ref):
    want = {}
    for ((fy, fx), bp), fl in zip(PATS, flips):
        fc = bp[1][1]; oth = BASE if fc == CLICK else CLICK
        a, b = (fc, oth) if not fl else (oth, fc)
        for i in range(3):
            for j in range(3):
                if (i == 1 and j == 1) or bp[i][j] == 3:
                    continue
                pos = (fy + (i-1)*8, fx + (j-1)*8)
                if pos not in cells:
                    continue
                tgt = a if bp[i][j] == 0 else b
                if pos in want and want[pos] != tgt:
                    return None
                want[pos] = tgt
    need = {}
    for p in cells:
        if skip_ref and p == REF:
            need[p] = 0
        else:
            need[p] = 1 if want.get(p, BASE) != BASE else 0
    clicks = []
    for x in xs:
        col = sorted([y for (y, xx) in cells if xx == x], reverse=True)
        flip = {y: 0 for y in col}
        for y in col:
            if (need[(y, x)] + flip[y]) % 2 == 1:
                clicks.append((x, y))
                flip[y] += 1
                if (y - 8, x) in flip:
                    flip[y - 8] += 1
    if not clicks:
        return None
    ch = clone(game)
    for (x, y) in clicks:
        fr = raw(ch, 6, {"x": x, "y": y})
        if fr.levels_completed >= level:
            return clicks
    return None

for flips in itertools.product((0, 1), repeat=len(PATS)):
    for skip_ref in (False, True):
        r = attempt(flips, skip_ref)
        if r:
            print(f"命中! flips={flips} skip_ref={skip_ref} {len(r)}击")
            for (x, y) in r:
                f = env.step(GameAction.ACTION6, {"x": x, "y": y})
            print(f"真机 levels={f.levels_completed} state={f.state.name}")
            if f.levels_completed >= level:
                sols["seqs"].append(r)
                json.dump(sols, open("ft09_solutions.json", "w"))
                print("已存 — ft09 全通!" if f.levels_completed >= 6 else "已存")
            raise SystemExit
print("32 组合全空")
