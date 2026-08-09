"""L6: 蓝图独立变换枚举(3位=无格做过滤) + lights-out + clone 验证。"""
import itertools, json, time
import numpy as np
import arc_agi
from arcengine import GameAction, ActionInput
from solve_ft09_all import blocks, clone, raw

OPS = [lambda x: x, lambda x: np.rot90(x, -1), lambda x: np.rot90(x, 2), lambda x: np.rot90(x, 1),
       np.fliplr, np.flipud, lambda x: np.array(x).T, lambda x: np.rot90(np.array(x).T, 2)]

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
cells = []
for (y, x) in cands:
    ch = clone(game)
    fr = raw(ch, 6, {"x": x, "y": y})
    if np.any(np.array(fr.frame[-1])[:63] != g0[:63]):
        cells.append((y, x))
cells.sort()
idx = {p: i for i, p in enumerate(cells)}
n = len(cells)
xs = sorted({x for _, x in cells})
ys = sorted({y for y, _ in cells})
PATS = []
for (fy, fx) in [(y, x) for y in ys for x in xs]:
    if (fy, fx) in cells:
        continue
    blk = g0[fy - 2:fy + 4, fx - 2:fx + 4]
    if blk.shape == (6, 6) and 0 in blk:
        PATS.append(((fy, fx), np.array([[int(blk[2*i][2*j]) for j in range(3)] for i in range(3)])))

BASE, CLICK = 11, 14

def clicks_for(need):
    out = []
    for x in xs:
        col = sorted([y for (y, xx) in cells if xx == x], reverse=True)
        flip = {y: 0 for y in col}
        for y in col:
            if (need[idx[(y, x)]] + flip[y]) % 2 == 1:
                out.append((x, y))
                flip[y] += 1
                if (y - 8, x) in flip:
                    flip[y - 8] += 1
    return out

t0 = time.time(); tested = 0; passed = 0
for ops in itertools.product(range(8), repeat=len(PATS)):
    want = {}
    ok = True
    for ((fy, fx), bp), oi in zip(PATS, ops):
        b = OPS[oi](bp)
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1:
                    continue
                pos = (fy + (i-1)*8, fx + (j-1)*8)
                exists = pos in idx
                v = b[i][j]
                if (v == 3) != (not exists):   # 3 当且仅当无格
                    ok = False; break
                if not exists:
                    continue
                tgt = CLICK if v == 0 else BASE
                if pos in want and want[pos] != tgt:
                    ok = False; break
                want[pos] = tgt
            if not ok: break
        if not ok: break
    if not ok:
        continue
    passed += 1
    need = [0]*n
    for p, tgt in want.items():
        need[idx[p]] = 1 if tgt != BASE else 0
    cl = clicks_for(need)
    if not cl:
        continue
    tested += 1
    ch = clone(game)
    win = False
    for (x, y) in cl:
        fr = raw(ch, 6, {"x": x, "y": y})
        if fr.levels_completed >= level:
            win = True; break
    if win:
        print(f"命中! ops={ops} {len(cl)}击 (谓词过{passed} 验证{tested} {time.time()-t0:.0f}s)")
        for (x, y) in cl:
            f = env.step(GameAction.ACTION6, {"x": x, "y": y})
        print(f"真机 levels={f.levels_completed} state={f.state.name}")
        if f.levels_completed >= level:
            sols["seqs"].append(cl)
            json.dump(sols, open("ft09_solutions.json", "w"))
            print("已存 — ft09 全通!" if f.levels_completed >= 6 else "已存")
        raise SystemExit
print(f"全空: 谓词过 {passed} 验证 {tested} ({time.time()-t0:.0f}s)")
