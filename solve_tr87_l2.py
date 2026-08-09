"""tr87 L2: 字典部分可查 -> 切分约束穷举。

答案 7 位 = 题面 4 符号的翻译串按序连接(段长 1-3)。可查符号的串固定,
未知段逐位枚举候选(每位环序实测)。全部组合在 clone 上验证, 命中后真机执行。
"""
import copy, itertools, json, sys, time
import numpy as np
import arc_agi
from arcengine import GameAction, ActionInput
from parse_tr87 import parse, show

ACTS = {i: getattr(GameAction, f"ACTION{i}") for i in range(1, 8)}

def raw(g, a):
    return g.perform_action(ActionInput(id=ACTS[a], data={}), raw=True)

def clone(g):
    clean = getattr(g, "_clean_levels", None)
    g._clean_levels = None; g2 = copy.deepcopy(g)
    g._clean_levels = clean; g2._clean_levels = clean
    return g2

arc = arc_agi.Arcade()
env = arc.make("tr87")
f = env.reset()
l1 = json.load(open("tr87_l1.json"))
for a in l1["l1_seq"]:
    f = env.step(ACTS[a])
assert f.levels_completed == 1
game = env._game
g = np.array(f.frame[-1])
pairs, prob, ans0, span = parse(g)
lut = {s[0]: list(d) for s, d in pairs if len(s) == 1}
N = len(ans0)
print(f"题面 {len(prob)} 符号, 答案 {N} 位")

# 每位候选环序(clone 上实测): 光标初始在位0
rings = []
base = clone(game)
for i in range(N):
    ch = clone(base)
    for _ in range(i):
        raw(ch, 4)
    ring = [ans0[i]]
    for _ in range(12):
        fr = raw(ch, 1)
        gg = np.array(fr.frame[-1])
        _, _, a2, _ = parse(gg)
        if a2[i] == ring[0]:
            break
        ring.append(a2[i])
    rings.append(ring)
print(f"各位环长: {[len(r) for r in rings]}")

# 切分方案: 题面 4 段, 每段 1-3, 总和 = N; 已知符号段长锁定为其串长
known = {j: lut[m] for j, m in enumerate(prob) if m in lut}
lens_options = []
for j, m in enumerate(prob):
    lens_options.append([len(known[j])] if j in known else [1, 2, 3])
splits = [c for c in itertools.product(*lens_options) if sum(c) == N]
print(f"切分方案 {len(splits)} 种: {splits}")

# 对每个切分: 固定已知段位, 未知位枚举各自环
t0 = time.time()
tried = 0
sol_ks = None
for sp in splits:
    slots = [None] * N        # 每位目标符号(None=未知)
    p = 0
    for j, L in enumerate(sp):
        if j in known:
            for t, mm in enumerate(known[j]):
                slots[p + t] = mm
        p += L
    unknown_idx = [i for i in range(N) if slots[i] is None]
    cand_lists = [rings[i] for i in unknown_idx]
    for combo in itertools.product(*cand_lists):
        tried += 1
        tgt = list(slots)
        for i, mm in zip(unknown_idx, combo):
            tgt[i] = mm
        # 目标必须在各位环内
        ks = []
        ok = True
        for i in range(N):
            if tgt[i] not in rings[i]:
                ok = False; break
            ks.append(rings[i].index(tgt[i]))
        if not ok:
            continue
        # 构造按键序列并 clone 验证
        seq = []
        last = max((i for i, k in enumerate(ks) if k > 0), default=-1)
        for i, k in enumerate(ks):
            n = len(rings[i])
            if k > 0:
                seq += [1] * k if k <= n - k else [2] * (n - k)
            if i < last:
                seq.append(4)
        ch = clone(game)
        done = False
        for a in seq:
            fr = raw(ch, a)
            if fr.levels_completed > 1:
                done = True; break
        if done:
            sol_ks = (ks, seq)
            print(f"命中! 组合 ks={ks} 步数={len(seq)} (尝试 {tried} 个组合, {time.time()-t0:.0f}s)")
            break
    if sol_ks:
        break

if not sol_ks:
    sys.exit(f"未找到 (尝试 {tried} 组合) — 假设有误")

ks, seq = sol_ks
for a in seq:
    f = env.step(ACTS[a])
print(f"真机执行后 levels={f.levels_completed} ({len(seq)}步 vs 人类58)")
if f.levels_completed >= 2:
    json.dump({"l2_seq": seq}, open("tr87_l2.json", "w"))
    print("L2 解已存")
