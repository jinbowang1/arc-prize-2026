"""ft09 v2: 花纹=邻域蓝图('.'位=点击态), 直读答案, clone 验证后真机执行。"""
import copy, itertools, json, os, time
import numpy as np
from collections import Counter, deque
import arc_agi
from arcengine import GameAction, ActionInput
from solve_ft09_all import blocks, clone, raw

def solve_level(game, g0, level):
    cands = blocks(g0)
    cells = {}
    for (y, x) in cands:
        ch = clone(game)
        fr = raw(ch, 6, {"x": x, "y": y})
        g1 = np.array(fr.frame[-1])
        if fr.levels_completed >= level:
            return [(x, y)]
        if np.any(g1[:63] != g0[:63]):
            cells[(y, x)] = True
    ys = sorted({y for y, _ in cells}); xs = sorted({x for _, x in cells})
    print(f"  可编辑格 {len(cells)}: y={ys} x={xs}", flush=True)
    # 格网 = 可编辑格坐标网 + 缺口(花纹位)
    grid = [(y, x) for y in ys for x in xs]
    clicks = []
    for (fy, fx) in grid:
        if (fy, fx) in cells:
            continue
        blk = g0[fy - 2:fy + 4, fx - 2:fx + 4]
        if blk.shape != (6, 6) or 0 not in blk:
            continue
        bp = [[int(blk[2 * i][2 * j]) for j in range(3)] for i in range(3)]
        print(f"  花纹@({fy},{fx}) 蓝图 {bp}", flush=True)
        dy = ys[1] - ys[0] if len(ys) > 1 else 8
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1:
                    continue
                ny, nx = fy + (i - 1) * dy, fx + (j - 1) * dy
                if (ny, nx) not in cells:
                    continue
                want_clicked = (bp[i][j] == 0)
                if want_clicked:
                    clicks.append((nx, ny))
    clicks = sorted(set(clicks))
    print(f"  蓝图要求点击 {len(clicks)} 格", flush=True)
    if not clicks:
        return None
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
print(f"重放 {len(sols['seqs'])} 关, levels={f.levels_completed}", flush=True)

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
