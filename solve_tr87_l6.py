"""L6: 两级翻译字典修正。谓词枚举 + clone 验证。"""
import copy, itertools, json, pickle, time
import numpy as np
import arc_agi
from arcengine import GameAction, ActionInput

ACTS = {i: getattr(GameAction, f"ACTION{i}") for i in range(1, 8)}
OPS = [lambda x: x, lambda x: np.rot90(x, -1), lambda x: np.rot90(x, 2), lambda x: np.rot90(x, 1),
       np.fliplr, np.flipud, lambda x: x.T, lambda x: np.rot90(x.T, 2)]
def raw(g, a): return g.perform_action(ActionInput(id=ACTS[a], data={}), raw=True)
def clone(g):
    clean = getattr(g, "_clean_levels", None)
    g._clean_levels = None; g2 = copy.deepcopy(g)
    g._clean_levels = clean; g2._clean_levels = clean
    return g2
def tup(m): return tuple(map(tuple, np.asarray(m).tolist()))

D = pickle.load(open("l6_data.pkl", "rb"))
exA, exB, rings = D["exA"], D["exB"], D["rings"]
# rings[i]: 框 i 的环, 每成员=符号列表; 框序 = 带t*4 + {0:A,1:77,2:7,3:B}


def canon(m):
    return min(tup(f(np.array(m))) for f in OPS)

def variants(m):
    return {tup(f(np.array(m))): fi for fi, f in enumerate(OPS)}

# 预计算: 带 t 的 A 框候选 (kA, 题面符号 i, 变换fi)
A_cand = [[] for _ in range(3)]
for t in range(3):
    for k, content in enumerate(rings[t * 4 + 0]):
        if len(content) != 1:
            continue
        cm = tup(content[0])
        for i, a in enumerate(exA):
            vs = variants(a)
            if cm in vs:
                A_cand[t].append((k, i, vs[cm]))
print(f"A 候选/带: {[len(c) for c in A_cand]}")

# 预计算: D2 可行性 feas[u][(p,b)] = [(k7,kB)]  (同变换 T')
from collections import defaultdict
feas = [defaultdict(list) for _ in range(3)]
all_pb = set()

def try_pb(u, p, b):
    key = (tup(p), tup(b))
    if key in feas[u]:
        return feas[u][key]
    out = []
    pv, bv = variants(p), variants(b)
    for k7, c7 in enumerate(rings[u * 4 + 2]):
        if len(c7) != 1: continue
        if tup(c7[0]) not in pv: continue
        for kB, cB in enumerate(rings[u * 4 + 3]):
            if len(cB) != 1: continue
            if tup(cB[0]) in bv:
                out.append((k7, kB))
    feas[u][key] = out
    return out

t0 = time.time()
sols_pred = []
for sigma in itertools.permutations(range(3)):   # 带 t 服务题面符号 sigma[t]
    for choices in itertools.product(*[
            [(t, kA, fi, k77, rev)
             for (kA, i, fi) in A_cand[t] if i == sigma[t]
             for k77 in range(len(rings[t * 4 + 1]))
             for rev in (0, 1)]
            for t in range(3)]):
      for layout in ("adj", "sym"):
        mid = [None] * 6
        ok = True
        for (t, kA, fi, k77, rev) in choices:
            i = sigma[t]
            seg = rings[t * 4 + 1][k77]
            if len(seg) != 2:
                ok = False; break
            p0, p1 = (seg[1], seg[0]) if rev else (seg[0], seg[1])
            pos = (2 * i, 2 * i + 1) if layout == "adj" else (i, 5 - i)
            mid[pos[0]], mid[pos[1]] = tup(p0), tup(p1)
        if not ok:
            continue
        # 查询集: (mid[j] -> exB[j]); dedupe
        need = {}
        for j in range(6):
            need.setdefault((canon(mid[j]), canon(exB[j])), None)
        if len(need) > 3:
            continue
        reqs = list(need.keys())
        for assign in itertools.permutations(range(3), len(reqs)):
            good = True
            opt_lists = []
            for (pq, bq), u in zip(reqs, assign):
                opts = try_pb(u, np.array(pq), np.array(bq))
                if not opts:
                    good = False; break
                opt_lists.append((u, opts))
            if good:
                for picks in itertools.product(*[o for _, o in opt_lists]):
                    ks = [0] * 12
                    for (t, kA, fi, k77, rev) in choices:
                        ks[t * 4 + 0], ks[t * 4 + 1] = kA, k77
                    for (u, _), (k7, kB) in zip(opt_lists, picks):
                        ks[u * 4 + 2], ks[u * 4 + 3] = k7, kB
                    sols_pred.append(ks)
print(f"谓词通过 {len(sols_pred)} 组合 ({time.time()-t0:.0f}s)")

# clone 验证
arc = arc_agi.Arcade()
env = arc.make("tr87")
f = env.reset()
S = json.load(open("tr87_solutions.json"))
for seq in S["seqs"]:
    for a in seq:
        f = env.step(ACTS[a])
game = env._game
level = f.levels_completed + 1

def keyseq(ks):
    seq = []
    nz = [i for i, k in enumerate(ks) if k > 0]
    if not nz:
        return []
    last = nz[-1]
    for i in range(12):
        k = ks[i]
        if k > 0:
            seq += [1] * k if k <= 7 - k else [2] * (7 - k)
        if i < last:
            seq.append(4)
    return seq

seen = set()
found = None
for ks in sols_pred:
    kk = tuple(ks)
    if kk in seen:
        continue
    seen.add(kk)
    seq = keyseq(ks)
    ch = clone(game)
    for a in seq:
        fr = raw(ch, a)
        if fr.levels_completed >= level:
            found = seq
            break
    if found:
        print(f"命中! ks={ks} {len(seq)}步")
        break
print(f"验证了 {len(seen)} 个不同组合")
if found:
    for a in found:
        f = env.step(ACTS[a])
    print(f"真机: levels={f.levels_completed} state={f.state.name} ({len(found)}步 vs 人类146)")
    if f.levels_completed >= level:
        S["seqs"].append(found)
        json.dump(S, open("tr87_solutions.json", "w"))
        print("已存 — tr87 全通!" if f.levels_completed >= 6 else "已存")
