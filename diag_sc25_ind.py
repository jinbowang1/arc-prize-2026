"""L3: 右边 行21-26 列47-58 有个物体, 中间是**色13 —— 与那道门同色**。
它是不是门的指示器? 什么能改变它?"""
import json
import numpy as np
from harness.env import Action, Game, action_space
from harness.run import _parse

PAL = " .:-=+*#%@$&XO<>"
sol = json.load(open("sc25_solutions.json"))
game, obs = Game.make("sc25")
for t in sol["seq"]:
    obs = game.act(_parse(t))
g0 = np.array(obs.grid)
IND = (20, 27, 45, 60)
print("指示器区 行20-27 列45-60:")
for r in range(IND[0], IND[1] + 1):
    print(f"  {r:>3} " + "".join(PAL[v % 16] for v in g0[r, IND[2]:IND[3]+1]))
print(f"  颜色: {sorted(set(g0[IND[0]:IND[1]+1, IND[2]:IND[3]+1].flatten().tolist()))}")
print(f"  色13 的格数: {int((g0[IND[0]:IND[1]+1, IND[2]:IND[3]+1] == 13).sum())}")

base = g0[IND[0]:IND[1]+1, IND[2]:IND[3]+1]
sp = action_space(list(obs.actions))
from harness.percept import analyze
sc = analyze(obs.grid)
acts = [Action.key(k) for k in sp["keys"]] + [Action.click(c, r) for (r, c) in sc.targets]
game.detect_lag(acts)
print("\n哪些动作能改变指示器:")
hit = 0
for a in acts:
    o = game.effect(a)
    if o.dead:
        continue
    sub = np.array(o.grid)[IND[0]:IND[1]+1, IND[2]:IND[3]+1]
    if not np.array_equal(sub, base):
        hit += 1
        d13 = int((sub == 13).sum())
        print(f"  {repr(a):<14} 改了指示器 | 色13 格数 {int((base==13).sum())} -> {d13}")
if not hit:
    print("  **没有任何单个动作能改变它**")
