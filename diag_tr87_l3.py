"""看 L3 字典/题面结构 + 验证 dst 串成员是否互为旋转族。"""
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
for seq in sols["seqs"]:
    for a in seq:
        f = env.step(ACTS[a])
print(f"levels={f.levels_completed}")
g = np.array(f.frame[-1])
pairs, prob, ans0, _ = parse(g)
print("=== L3 字典 ===")
for i, (s, d) in enumerate(pairs):
    print(f"  对{i+1}: src串长{len(s)} dst串长{len(d)}")
    print(f"    src: {' '.join(show(m) for m in s)}")
    print(f"    dst: {' '.join(show(m) for m in d)}")
print("=== L3 题面 ===")
for m in prob:
    print("  ", show(m))
print(f"=== L3 答案区 {len(ans0)} 位(初始) ===")

print("\n=== L2 dst 串成员旋转族验证 ===")
env2 = arc_agi.Arcade().make("tr87")
f2 = env2.reset()
for a in sols["seqs"][0]:
    f2 = env2.step(ACTS[a])
g2 = np.array(f2.frame[-1])
pairs2, _, _, _ = parse(g2)
for i, (s, d) in enumerate(pairs2):
    if len(d) < 2:
        continue
    rel01 = [op for op, fn in OPS.items() if np.array_equal(fn(np.array(d[0])), np.array(d[1]))]
    rels = [(a_, b_, [op for op, fn in OPS.items() if np.array_equal(fn(np.array(d[a_])), np.array(d[b_]))])
            for a_ in range(len(d)) for b_ in range(len(d)) if a_ < b_]
    print(f"  L2对{i+1}: " + "; ".join(f"串{a_+1}->串{b_+1}: {r or '非变换关系'}" for a_, b_, r in rels))
