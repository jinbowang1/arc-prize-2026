"""覆盖诊断: B框环 vs 答案符号; 7框环 vs 77符号; A框环 vs 题面。"""
import pickle
import numpy as np
from parse_tr87 import show

OPS = [lambda x: x, lambda x: np.rot90(x, -1), lambda x: np.rot90(x, 2), lambda x: np.rot90(x, 1),
       np.fliplr, np.flipud, lambda x: x.T, lambda x: np.rot90(x.T, 2)]
def tup(m): return tuple(map(tuple, np.asarray(m).tolist()))
def variants(m): return {tup(f(np.array(m))) for f in OPS}

D = pickle.load(open("l6_data.pkl", "rb"))
exA, exB, rings = D["exA"], D["exB"], D["rings"]

print("— 答案符号 vs 各带 B 框环:")
for j, b in enumerate(exB):
    vs = variants(b)
    hit = []
    for u in range(3):
        for k, c in enumerate(rings[u * 4 + 3]):
            if len(c) == 1 and tup(c[0]) in vs:
                hit.append(f"带{u+1}.B[k={k}]")
    print(f"  答案{j+1}: {hit or '无匹配!'}")

print("— 题面符号 vs 各带 A 框环:")
for i, a in enumerate(exA):
    vs = variants(a)
    hit = []
    for t in range(3):
        for k, c in enumerate(rings[t * 4 + 0]):
            if len(c) == 1 and tup(c[0]) in vs:
                hit.append(f"带{t+1}.A[k={k}]")
    print(f"  题{i+1}: {hit or '无!'}")

print("— 7 框环成员能否在 77 符号集(全带全环)中找到变换匹配:")
S77 = []
for t in range(3):
    for k, c in enumerate(rings[t * 4 + 1]):
        for x in c:
            S77.append(tup(x))
S77v = set()
for x in S77:
    S77v |= variants(x)
for u in range(3):
    ok = sum(1 for c in rings[u * 4 + 2] if len(c) == 1 and tup(c[0]) in S77v)
    print(f"  带{u+1}.7: {ok}/{len(rings[u*4+2])} 成员可由 77 符号变换而来")
