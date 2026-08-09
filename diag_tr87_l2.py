"""诊断 L2: 每位环内容 vs 候选段符号, 看约束卡在哪。"""
import copy, json
import numpy as np
import arc_agi
from arcengine import GameAction, ActionInput
from parse_tr87 import parse, show
from solve_tr87 import seg_candidates, clone, raw, tup, ACTS

arc = arc_agi.Arcade()
env = arc.make("tr87")
f = env.reset()
sols = json.load(open("tr87_solutions.json"))
for a in sols["seqs"][0]:
    f = env.step(ACTS[a])
game = env._game
g = np.array(f.frame[-1])
pairs, prob, ans0, _ = parse(g)
N = len(ans0)

rings = []
for i in range(N):
    ch = clone(game)
    for _ in range(i):
        raw(ch, 4)
    ring = [tup(ans0[i])]
    for _ in range(12):
        fr = raw(ch, 1)
        _, _, a2, _ = parse(np.array(fr.frame[-1]))
        if tup(a2[i]) == ring[0]:
            break
        ring.append(tup(a2[i]))
    rings.append(ring)

allring = set()
for r in rings:
    allring |= set(r)
print(f"环符号全集大小: {len(allring)}; 各位环长 {[len(r) for r in rings]}")
same = all(set(r) == set(rings[0]) for r in rings)
print(f"各位环符号集合是否相同: {same}")

for j, m in enumerate(prob):
    print(f"\n题面符号{j+1}: {show(m)}")
    for c in seg_candidates(m, pairs):
        marks = ["∈环" if tup(x) in allring else "∉环" for x in c]
        print("   候选段:", " | ".join(f"{show(x)} {k}" for x, k in zip(c, marks)))

print("\n=== 逐位环检查(段顺序=题面顺序) ===")
import itertools
cands = [seg_candidates(m, pairs) for m in prob]
n_ok = 0
for combo in itertools.product(*cands):
    target = [x for seg in combo for x in seg]
    if len(target) != N:
        continue
    bad = [(i, show(target[i])) for i in range(N) if tup(target[i]) not in rings[i]]
    if not bad:
        n_ok += 1
        print(f"  可行组合#{n_ok}: " + " ".join(show(t)[:11] for t in target))
    elif len(bad) <= 2:
        print(f"  近可行(卡{len(bad)}位): " + "; ".join(f"位{i}={s[:11]}" for i, s in bad))
print(f"逐位全可行组合数: {n_ok}")
print("\n=== 各位环内容 ===")
for i, r in enumerate(rings):
    print(f"位{i}: " + " ".join(show(np.array(m))[:11] for m in r))
