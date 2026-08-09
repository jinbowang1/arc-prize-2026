"""tr87 L2 兜底: 从初始答案区出发, 按改动位数 k 递增穷举(每位 7 候选环), clone 验证。"""
import copy, itertools, json, time
import numpy as np
import arc_agi
from arcengine import GameAction, ActionInput
from parse_tr87 import parse, show
from solve_tr87 import clone, raw, tup, ACTS

arc = arc_agi.Arcade()
env = arc.make("tr87")
f = env.reset()
sols = json.load(open("tr87_solutions.json"))
for seq in sols["seqs"]:
    for a in seq:
        f = env.step(ACTS[a])
game = env._game
level = f.levels_completed + 1
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

def keyseq(ks):
    seq = []
    idx = [i for i, k in enumerate(ks) if k > 0]
    if not idx:
        return []
    last = idx[-1]
    for i, k in enumerate(ks):
        n = len(rings[i])
        if k > 0:
            seq += [1] * k if k <= n - k else [2] * (n - k)
        if i < last:
            seq.append(4)
    return seq

t0 = time.time()
found = None
for kk in range(1, 5):
    cnt = 0
    for pos in itertools.combinations(range(N), kk):
        for vals in itertools.product(*[range(1, len(rings[i])) for i in pos]):
            ks = [0] * N
            for i, v in zip(pos, vals):
                ks[i] = v
            seq = keyseq(ks)
            ch = clone(game)
            win = False
            for a in seq:
                fr = raw(ch, a)
                if fr.levels_completed >= level:
                    win = True; break
            cnt += 1
            if win:
                found = seq
                print(f"命中! 改{kk}位 {pos} vals={vals}, {len(seq)}步 ({time.time()-t0:.0f}s)")
                break
        if found: break
    print(f"k={kk}: 试了 {cnt} 组合, 累计 {time.time()-t0:.0f}s")
    if found: break

if found:
    for a in found:
        f = env.step(ACTS[a])
    print(f"真机: levels={f.levels_completed} ({len(found)}步 vs 人类58)")
    if f.levels_completed >= level:
        sols["seqs"].append(found)
        json.dump(sols, open("tr87_solutions.json", "w"))
        print("已存 tr87_solutions.json")
