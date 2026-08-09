"""L3 真实答案提取 + 三关联合规则拟合素材。"""
import json
import numpy as np
import arc_agi
from arcengine import ActionInput, GameAction
from parse_tr87 import parse, show
from solve_tr87 import clone, raw, tup, ACTS, OPS

arc = arc_agi.Arcade()
env = arc.make("tr87")
f = env.reset()
sols = json.load(open("tr87_solutions.json"))
for seq in sols["seqs"][:2]:
    for a in seq:
        f = env.step(ACTS[a])
game = env._game
g = np.array(f.frame[-1])
pairs, prob, ans0, _ = parse(g)

seq3 = sols["seqs"][2]
for a in seq3[:-1]:
    raw(game, a)
fr = raw(game, seq3[-1])
g_fin = np.array(fr.frame[0])
_, _, ans_fin, _ = parse(g_fin)

print("=== L3 真实答案(7位) ===")
for m in ans_fin:
    print("  ", show(m))

print("\n=== 答案[i] 与 dst 成员全变换匹配 ===")
for i, a in enumerate(ans_fin):
    am = np.array(a)
    hits = []
    for j, (s, d) in enumerate(pairs):
        for t, dd in enumerate(d):
            for op, fn in OPS.items():
                if np.array_equal(fn(np.array(dd)), am):
                    hits.append(f"对{j+1}.{t+1}:{op}")
    print(f"  位{i+1}: {hits or '字典外!'}")

print("\n=== 题面[j] 与 src 成员全变换匹配(逐符号) ===")
for j, m in enumerate(prob):
    pm = np.array(m)
    hits = []
    for i, (s, d) in enumerate(pairs):
        for t, ss in enumerate(s):
            for op, fn in OPS.items():
                if np.array_equal(fn(np.array(ss)), pm):
                    hits.append(f"对{i+1}.{t+1}:{op}")
    print(f"  题{j+1}: {hits or '无'}")
