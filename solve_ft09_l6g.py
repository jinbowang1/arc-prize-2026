"""L6: 中心对称/反对称态全枚举(2^11 x2)。"""
import itertools, json, time
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
CY, CX = 28, 30
pairs = []
used = set()
for p in cells:
    if p in used:
        continue
    q = (2 * CY - p[0], 2 * CX - p[1])
    assert q in idx, f"{p} 无镜像"
    used.add(p); used.add(q)
    pairs.append((p, q))
print(f"{len(pairs)} 对")

def clicks_for(need):
    out = []
    for x in xs:
        col = sorted([y for (y, xx) in cells if xx == x], reverse=True)
        flip = {y: 0 for y in col}
        for y in col:
            if (need[idx[(y, x)]] + flip[y]) % 2 == 1:
                out.append((x, y)); flip[y] += 1
                if (y - 8, x) in flip:
                    flip[y - 8] += 1
    return out

t0 = time.time()
tested = 0
for anti in (0, 1):
    for bits in itertools.product((0, 1), repeat=len(pairs)):
        need = [0] * n
        for (p, q), b in zip(pairs, bits):
            need[idx[p]] = b
            need[idx[q]] = b ^ anti
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
            print(f"命中! anti={anti} bits={bits} {len(cl)}击 ({tested}试 {time.time()-t0:.0f}s)")
            for (x, y) in cl:
                f = env.step(GameAction.ACTION6, {"x": x, "y": y})
            print(f"真机 levels={f.levels_completed} state={f.state.name}")
            if f.levels_completed >= level:
                sols["seqs"].append(cl)
                json.dump(sols, open("ft09_solutions.json", "w"))
                print("已存 — ft09 全通!" if f.levels_completed >= 6 else "已存")
            raise SystemExit
    print(f"anti={anti} 空 ({tested}试 {time.time()-t0:.0f}s)", flush=True)
print("全空")
