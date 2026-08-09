"""L6: 重放格雷码命中路径(整段)真机过关。"""
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

c_hit = int("0110010011101100110010", 2)
b = c_hit
g_inv = 0
while b:
    g_inv ^= b
    b >>= 1
gi = g_inv
assert gi ^ (gi >> 1) == c_hit
SEG = 100
s0 = (gi - 1) // SEG * SEG
print(f"命中 gi={gi}, 段起点 {s0}")

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

path = clicks_for(need_of(s0 ^ (s0 >> 1)))
for t in range(s0 + 1, gi + 1):
    j = (t & -t).bit_length() - 1
    path.append((cells[j][1], cells[j][0]))
print(f"整段路径 {len(path)} 击")

ch = clone(game)
win_at = None
for k, (x, y) in enumerate(path):
    fr = raw(ch, 6, {"x": x, "y": y})
    if fr.levels_completed >= level:
        win_at = k + 1; break
print(f"clone 复核: {'WIN@' + str(win_at) if win_at else '未过?!'}")
if win_at:
    path = path[:win_at]
    for (x, y) in path:
        f = env.step(GameAction.ACTION6, {"x": x, "y": y})
    print(f"真机 levels={f.levels_completed} state={f.state.name} ({len(path)}击 vs 人类37)")
    if f.levels_completed >= level:
        sols["seqs"].append(path)
        json.dump(sols, open("ft09_solutions.json", "w"))
        print("已存 — ft09 全通(待优化步数)!")
