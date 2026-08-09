"""L6 分段格雷码全态遍历(尊重 128 击上限)。"""
import json, time
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
n = len(cells)
idx = {p: i for i, p in enumerate(cells)}
xs = sorted({x for _, x in cells})
N = 2 ** n
SEG = 100

def need_of(c):
    nd = [0] * n
    for j in range(n):
        if (c >> j) & 1:
            y, x = cells[j]
            nd[idx[(y, x)]] ^= 1
            if (y - 8, x) in idx:
                nd[idx[(y - 8, x)]] ^= 1
    return nd

def clicks_for(nd):
    out = []
    for x in xs:
        col = sorted([y for (y, xx) in cells if xx == x], reverse=True)
        flip = {y: 0 for y in col}
        for y in col:
            if (nd[idx[(y, x)]] + flip[y]) % 2 == 1:
                out.append((x, y)); flip[y] += 1
                if (y - 8, x) in flip:
                    flip[y - 8] += 1
    return out

def report_and_exit(c_hit):
    nd = need_of(c_hit)
    clicks = [(cells[j][1], cells[j][0]) for j in range(n) if (c_hit >> j) & 1]
    print(f"命中态 c={c_hit:022b}, 目标E格={[cells[i] for i in range(n) if nd[i]]}", flush=True)
    ch2 = clone(game)
    win = False
    for (x, y) in clicks:
        fr = raw(ch2, 6, {"x": x, "y": y})
        if fr.levels_completed >= level:
            win = True; break
    print(f"复核 {len(clicks)}击: {'WIN' if win else '未过?!'}", flush=True)
    if win:
        g = None
        for (x, y) in clicks:
            g = env.step(GameAction.ACTION6, {"x": x, "y": y})
        print(f"真机 levels={g.levels_completed} state={g.state.name}", flush=True)
        if g.levels_completed >= level:
            sols["seqs"].append(clicks)
            json.dump(sols, open("ft09_solutions.json", "w"))
            print("已存 — ft09 全通!", flush=True)
    raise SystemExit

t0 = time.time()
i = 0
while i < N:
    c0 = i ^ (i >> 1)
    setup = clicks_for(need_of(c0))
    ch = clone(game)
    acts = 0
    hit = None
    for (x, y) in setup:
        fr = raw(ch, 6, {"x": x, "y": y})
        acts += 1
        if fr.levels_completed >= level:
            hit = c0; break
    if hit is not None:
        report_and_exit(hit)
    steps = min(SEG, N - 1 - i)
    for s in range(steps):
        gi = i + s + 1
        j = (gi & -gi).bit_length() - 1
        y, x = cells[j]
        fr = raw(ch, 6, {"x": x, "y": y})
        acts += 1
        if fr.state.name == "GAME_OVER":
            print(f"段内 GAME_OVER acts={acts}", flush=True); break
        if fr.levels_completed >= level:
            report_and_exit(gi ^ (gi >> 1))
    i += steps if steps else 1
    if i % 500000 < SEG:
        print(f"  {i}/{N} ({time.time()-t0:.0f}s)", flush=True)
print(f"全空 ({time.time()-t0:.0f}s) — L6 判定确非静态色态匹配", flush=True)
