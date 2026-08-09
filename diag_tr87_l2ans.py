"""L2 真实答案 vs 题面/字典: 全变换匹配。"""
import copy, json
import numpy as np
import arc_agi
from arcengine import ActionInput, GameAction
from parse_tr87 import parse, show
from solve_tr87 import clone, raw, tup, ACTS, OPS

arc = arc_agi.Arcade()
env = arc.make("tr87")
f = env.reset()
sols = json.load(open("tr87_solutions.json"))
for seq in sols["seqs"][:1]:
    for a in seq:
        f = env.step(ACTS[a])
game = env._game
g = np.array(f.frame[-1])
pairs, prob, ans0, _ = parse(g)

# 重放 L2 解到倒数第二步, 再单步拿判定前最终帧
seq2 = sols["seqs"][1]
for a in seq2[:-1]:
    raw(game, a)
fr = raw(game, seq2[-1])
g_fin = np.array(fr.frame[0])
_, _, ans_fin, _ = parse(g_fin)

print("=== L2 字典 ===")
for i, (s, d) in enumerate(pairs):
    print(f"  对{i+1}: {' '.join(show(m) for m in s)}  ->  {' '.join(show(m) for m in d)}")
print("=== L2 题面 ===")
for m in prob:
    print("  ", show(m))
print("=== L2 真实答案(7位) ===")
for m in ans_fin:
    print("  ", show(m))

print("\n=== 答案[i] 与字典 dst 成员的全变换匹配 ===")
for i, a in enumerate(ans_fin):
    am = np.array(a)
    hits = []
    for j, (s, d) in enumerate(pairs):
        for t, dd in enumerate(d):
            for op, fn in OPS.items():
                if np.array_equal(fn(np.array(dd)), am):
                    hits.append(f"对{j+1}串位{t+1}:{op}")
    print(f"  位{i+1}: {hits or '字典外符号!'}")

print("\n=== 题面[j] 与字典 src 的全变换匹配 ===")
for j, m in enumerate(prob):
    pm = np.array(m)
    hits = []
    for i, (s, d) in enumerate(pairs):
        for op, fn in OPS.items():
            if np.array_equal(fn(np.array(s[0])), pm):
                hits.append(f"对{i+1}:{op}(串长{len(d)})")
    print(f"  题{j+1}: {hits}")
