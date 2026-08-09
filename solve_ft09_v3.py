"""ft09 v3: 蓝图 + 每花纹独立变换枚举(解决方向性), 一致性过滤 + clone 验证。"""
import copy, itertools, json, os
import numpy as np
import arc_agi
from arcengine import GameAction, ActionInput
from solve_ft09_all import blocks, clone, raw

OPS = [lambda x: x, lambda x: np.rot90(x, -1), lambda x: np.rot90(x, 2), lambda x: np.rot90(x, 1),
       np.fliplr, np.flipud, lambda x: x.T, lambda x: np.rot90(x.T, 2)]

def solve_level(game, g0, level):
    cands = blocks(g0)
    cells = {}
    base_col = click_col = None
    for (y, x) in cands:
        ch = clone(game)
        fr = raw(ch, 6, {"x": x, "y": y})
        if fr.levels_completed >= level:
            return [(x, y)]
        g1 = np.array(fr.frame[-1])
        if np.any(g1[:63] != g0[:63]):
            cells[(y, x)] = True
            if base_col is None:
                d = np.argwhere(g1[:63] != g0[:63])
                base_col = int(g0[d[0][0], d[0][1]]); click_col = int(g1[d[0][0], d[0][1]])
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
    print(f"  可编辑 {len(cells)} 花纹 {len(pats)} 原色{base_col} 点击色{click_col}", flush=True)
    if not pats:
        return None
    want = {}
    for (fy, fx, bp) in pats:
        fc = int(bp[1][1])
        zero_clicks = (fc == click_col)   # 中心=点击色 -> '.'位点击; 中心=原色 -> '2'位点击
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1:
                    continue
                pos = (fy + (i - 1) * dy, fx + (j - 1) * dy)
                if pos not in cells:
                    continue
                w = (bp[i][j] == 0) if zero_clicks else (bp[i][j] != 0)
                if pos in want and want[pos] != w:
                    print(f"  冲突@{pos}"); return None
                want[pos] = w
    clicks = sorted((x, y) for (y, x), w in want.items() if w)
    print(f"  蓝图点击 {len(clicks)} 格", flush=True)
    ch = clone(game)
    win = False
    for (x, y) in clicks:
        fr = raw(ch, 6, {"x": x, "y": y})
        if fr.levels_completed >= level:
            win = True; break
    return clicks if win else None

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
