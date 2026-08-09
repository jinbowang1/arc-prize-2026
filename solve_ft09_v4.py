"""ft09 v4: 三态环 + 蓝图(0->中心色, 2->另一色, 歧义枚举) + clone 验证。"""
import copy, itertools, json, os
import numpy as np
import arc_agi
from arcengine import GameAction, ActionInput
from solve_ft09_all import blocks, clone, raw

def solve_level(game, g0, level):
    cands = blocks(g0)
    cells = {}
    ring_cols = None
    for (y, x) in cands:
        ch = clone(game)
        fr = raw(ch, 6, {"x": x, "y": y})
        if fr.levels_completed >= level:
            return [(x, y)]
        g1 = np.array(fr.frame[-1])
        if np.any(g1[:63] != g0[:63]):
            cells[(y, x)] = int(g0[y, x])
            if ring_cols is None:
                ring_cols = [int(g0[y, x]), int(g1[y, x])]
                prev = g1
                for _ in range(5):
                    fr = raw(ch, 6, {"x": x, "y": y})
                    gg = np.array(fr.frame[-1])
                    c = int(gg[y, x])
                    if c == ring_cols[0]:
                        break
                    ring_cols.append(c)
                    prev = gg
    ys = sorted({y for y, _ in cells}); xs = sorted({x for _, x in cells})
    dy = ys[1] - ys[0] if len(ys) > 1 else 8
    pats = []
    for (fy, fx) in [(y, x) for y in ys for x in xs]:
        if (fy, fx) in cells:
            continue
        blk = g0[fy - 2:fy + 4, fx - 2:fx + 4]
        if blk.shape != (6, 6) or 0 not in blk:
            continue
        bp = np.array([[int(blk[2 * i][2 * j]) for j in range(3)] for i in range(3)])
        pats.append((fy, fx, bp))
    print(f"  可编辑 {len(cells)} 花纹 {len(pats)} 环 {ring_cols}", flush=True)
    if not pats:
        return None
    R = len(ring_cols)
    # 每花纹: 0->中心色(需在环内), 2->另一色(环内非中心色, 枚举)
    opts = []
    for (fy, fx, bp) in pats:
        fc = int(bp[1][1])
        if fc not in ring_cols:
            print(f"  花纹中心色 {fc} 不在环 {ring_cols}!"); return None
        others = [c for c in ring_cols if c != fc]
        opts.append(others)
    for choice in itertools.product(*opts):
        want = {}
        ok = True
        for (fy, fx, bp), oth in zip(pats, choice):
            fc = int(bp[1][1])
            for i in range(3):
                for j in range(3):
                    if i == 1 and j == 1:
                        continue
                    pos = (fy + (i - 1) * dy, fx + (j - 1) * dy)
                    if pos not in cells:
                        continue
                    tgt = fc if bp[i][j] == 0 else oth
                    if pos in want and want[pos] != tgt:
                        ok = False; break
                    want[pos] = tgt
                if not ok: break
            if not ok: break
        if not ok:
            continue
        clicks = []
        for (y, x), tgt in want.items():
            cur_i = ring_cols.index(cells[(y, x)])
            k = (ring_cols.index(tgt) - cur_i) % R
            clicks += [(x, y)] * k
        if not clicks:
            continue
        clicks.sort()
        ch = clone(game)
        win = False
        for (x, y) in clicks:
            fr = raw(ch, 6, {"x": x, "y": y})
            if fr.levels_completed >= level:
                win = True; break
        if win:
            print(f"  命中 choice={choice} {len(clicks)}击", flush=True)
            return clicks
    return None

HUMAN = [43, 12, 23, 28, 65, 37]
arc = arc_agi.Arcade()
env = arc.make("ft09")
f = env.reset()
game = env._game
sols = json.load(open("ft09_solutions.json")) if os.path.exists("ft09_solutions.json") else {"seqs": []}
for seq in sols["seqs"]:
    for (x, y) in seq:
        f = env.step(GameAction.ACTION6, {"x": x, "y": y})
print(f"重放 {len(sols['seqs'])} 关 levels={f.levels_completed}", flush=True)
while f.levels_completed < f.win_levels:
    lvl = f.levels_completed + 1
    print(f"— L{lvl} (人类{HUMAN[lvl-1]})", flush=True)
    g0 = np.array(f.frame[-1])
    seq = solve_level(game, g0, lvl)
    if seq is None:
        print("未解, 停"); break
    for (x, y) in seq:
        f = env.step(GameAction.ACTION6, {"x": x, "y": y})
    if f.levels_completed >= lvl:
        sols["seqs"].append(seq)
        json.dump(sols, open("ft09_solutions.json", "w"))
        print(f"  L{lvl} ✓ {len(seq)}击 state={f.state.name}", flush=True)
    else:
        print("  真机未过?!"); break
print(f"最终 {f.levels_completed}/{f.win_levels} state={f.state.name} 各关={[len(s) for s in sols['seqs']]}")
