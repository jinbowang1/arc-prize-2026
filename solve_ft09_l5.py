"""L5: 先拨 6-块开关到蓝图态, 再按 want 补普通格, clone 验证后真机。"""
import json
import numpy as np
import arc_agi
from arcengine import GameAction, ActionInput
from solve_ft09_all import clone, raw

arc = arc_agi.Arcade()
env = arc.make("ft09")
f = env.reset()
sols = json.load(open("ft09_solutions.json"))
for seq in sols["seqs"]:
    for (x, y) in seq:
        f = env.step(GameAction.ACTION6, {"x": x, "y": y})
game = env._game
level = f.levels_completed + 1
g0 = np.array(f.frame[-1])

# want(来自诊断, 全部格的目标色): 由 8 花纹得出
WANT = {}
PATS = [((6,16),[[3,3,3],[3,14,0],[3,0,2]]), ((14,40),[[0,3,3],[2,15,3],[0,2,0]]),
        ((30,8),[[3,2,0],[3,15,2],[3,2,0]]), ((30,40),[[2,0,2],[0,14,0],[2,0,2]]),
        ((30,56),[[2,0,3],[0,14,3],[2,0,3]]), ((38,24),[[0,2,0],[2,14,2],[0,3,0]]),
        ((46,24),[[2,3,2],[0,14,0],[2,0,2]]), ((54,48),[[2,0,3],[0,14,3],[3,3,3]])]
for (fy, fx), bp in PATS:
    fc = bp[1][1]; oth = 15 if fc == 14 else 14
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1 or bp[i][j] == 3:
                continue
            WANT[(fy + (i-1)*8, fx + (j-1)*8)] = fc if bp[i][j] == 0 else oth

SIX = [(14, 24), (30, 24), (46, 40)]
CELLS = [k for k in WANT if k not in SIX]

def six_state(g, y, x):
    return 15 if 15 in set(g[y-2:y+4, x-2:x+4].flatten().tolist()) else 14

ch = clone(game)
clicks = []
# 1. 开关到位
for (y, x) in SIX:
    g = np.array(ch.frames[-1].frame[-1]) if False else None
for (y, x) in SIX:
    fr = raw(ch, 3)  # 无操作拿帧? ACTION3 可能有副作用 — 改用记录帧
# 重来: 用 raw 返回帧流跟踪
ch = clone(game)
cur = g0.copy()
for (y, x) in SIX:
    if (y, x) in WANT and six_state(cur, y, x) != WANT[(y, x)]:
        fr = raw(ch, 6, {"x": x, "y": y})
        cur = np.array(fr.frame[-1])
        clicks.append((x, y))
# 2. 普通格补差
win = False
for (y, x) in sorted(CELLS):
    tgt = WANT[(y, x)]
    if int(cur[y, x]) != tgt:
        fr = raw(ch, 6, {"x": x, "y": y})
        cur = np.array(fr.frame[-1])
        clicks.append((x, y))
        if fr.levels_completed >= level:
            win = True; break
print(f"clone: {len(clicks)}击 {'WIN' if win else '未过'}")
if win:
    for (x, y) in clicks:
        f = env.step(GameAction.ACTION6, {"x": x, "y": y})
    print(f"真机 levels={f.levels_completed}")
    if f.levels_completed >= level:
        sols["seqs"].append(clicks)
        json.dump(sols, open("ft09_solutions.json", "w"))
        print("已存")
