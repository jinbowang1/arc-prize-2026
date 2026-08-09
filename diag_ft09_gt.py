"""ft09: L1/L2 通关前最终画面(ground truth)与初始画面对照。"""
import json
import numpy as np
import arc_agi
from arcengine import GameAction, ActionInput
from solve_ft09_all import clone, raw

CH = ".123456789ABCDEF"
def show(g, r0, r1):
    for r in range(r0, r1):
        print(f"{r:>3} " + "".join(CH[v] for v in g[r]))

sols = json.load(open("ft09_solutions.json"))
arc = arc_agi.Arcade()
env = arc.make("ft09")
f = env.reset()
game = env._game

for li, seq in enumerate(sols["seqs"]):
    g_before = np.array(f.frame[-1])
    for (x, y) in seq[:-1]:
        f = env.step(GameAction.ACTION6, {"x": x, "y": y})
    # 最后一击在 clone 上打, 取判定前帧
    ch = clone(game)
    fr = raw(ch, 6, {"x": seq[-1][0], "y": seq[-1][1]})
    g_final = np.array(fr.frame[0])
    print(f"===== L{li+1} 初始 =====")
    show(g_before, 0, 62)
    print(f"===== L{li+1} 通关时(判定前帧) =====")
    show(g_final, 0, 62)
    f = env.step(GameAction.ACTION6, {"x": seq[-1][0], "y": seq[-1][1]})
    print(f"(真机 levels={f.levels_completed})")
