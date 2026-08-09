"""看当前关(重放后)的结构: 题面/答案位数/字典串长。"""
import json
import numpy as np
import arc_agi
from arcengine import GameAction
from parse_tr87 import parse, show

ACTS = {i: getattr(GameAction, f"ACTION{i}") for i in range(1, 8)}
arc = arc_agi.Arcade()
env = arc.make("tr87")
f = env.reset()
sols = json.load(open("tr87_solutions.json"))
for seq in sols["seqs"]:
    for a in seq:
        f = env.step(ACTS[a])
print(f"levels={f.levels_completed}/{f.win_levels} 各关步数={[len(s) for s in sols['seqs']]}")
g = np.array(f.frame[-1])
pairs, prob, ans0, _ = parse(g)
print(f"L{f.levels_completed+1}: 题面{len(prob)}符号 答案{len(ans0)}位")
for i, (s, d) in enumerate(pairs):
    print(f"  对{i+1}: src{len(s)} -> dst{len(d)}")
CH = ".123456789ABCDEF"
print("\n帧概览(每4行取1行):")
for r in range(0, 64, 4):
    print(f"{r:>3}  " + "".join(CH[v] for v in g[r]))
