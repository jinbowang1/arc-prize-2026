"""L3 决定性实验: 用**真正的语义状态**做指纹, BFS 穷尽。

新认识(干净实验): 画布每格是**二值**的(初始色 <-> 14), 与构型无关。
所以语义状态 = (9 格的二值, 方块位置) = 2^9 x 42 ≈ **21504**, 小得能穷尽。
而当前那个跑用的是"画布区字节 + 构型区字节", 已经见过 52422 个状态 ——
**指纹比语义状态细得多**, 带着像素细节, 白搜了一堆等价状态。

两种结局都有信息:
  找到解  -> L3 破
  **队列穷尽仍无解** -> 硬结论: 过关条件**不在 (画布 x 方块位置) 空间内**,
                       必有第三个维度, 那时该去找它而不是加预算。
"""
from __future__ import annotations

import json
import time
from collections import deque

import numpy as np

from harness.env import Action, Game
from harness.run import _parse

R, C = [49, 54, 59], [24, 29, 34]
sol = json.load(open("sc25_solutions.json"))
game, obs = Game.make("sc25")
for t in sol["seq"]:
    obs = game.act(_parse(t))
L0 = obs.level
g0 = np.array(obs.grid)
INIT = [[int(g0[R[i] + 1, C[j] + 1]) for j in range(3)] for i in range(3)]
print(f"L{L0+1} 画布初始色 {INIT} (每格在 初始色 <-> 14 之间切换)", flush=True)

CLICKS = [Action.click(C[j] + 1, R[i] + 1) for i in range(3) for j in range(3)]
KEYS = [Action.key(k) for k in (1, 2, 3, 4)]
ACTS = CLICKS + KEYS

def sem(g):
    """语义状态 = (9 格是否已变成 14, 方块中心)"""
    bits = tuple(1 if int(g[R[i] + 1, C[j] + 1]) == 14 else 0
                 for i in range(3) for j in range(3))
    # 🚨只在方块可达区找, 别扫全屏。方块可达 行11.5-31.5 x 列28.5-44.5,
    # 取 (10..34, 26..46) 足够覆盖。全屏 argwhere 是 4096 格, 这里 480 格 ——
    # 30 万个状态就是 12 亿次 vs 1.4 亿次扫描, sem() 本来是主要瓶颈。
    sub = g[10:34, 26:46]
    m = ((sub == 9) | (sub == 10))
    c = np.argwhere(m)
    pos = (round(float(c[:, 0].mean()) + 10, 1),
           round(float(c[:, 1].mean()) + 26, 1)) if len(c) else None
    return (bits, pos)

t0 = time.time()
start = sem(g0)
seen = {(start, obs.pending)}
q = deque([([], game.fork(), obs)])
expanded = deepest = 0
while q:
    seq, node, ob = q.popleft()
    expanded += 1
    deepest = max(deepest, len(seq))
    for a in ACTS:
        ch = node.fork()
        o = ch.act(a)
        if o.dead:
            continue
        if o.level > L0:
            print(f"🏆 过关! {len(seq)+1} 步 | 扩展 {expanded} | {time.time()-t0:.0f}s", flush=True)
            print("解:", [str(x) for x in seq + [a]], flush=True)
            raise SystemExit(0)
        p = ch.fork().act(a)                     # 补步查 level
        if not p.dead and p.level > L0:
            print(f"🏆 过关(补步)! {len(seq)+2} 步 | 扩展 {expanded}", flush=True)
            print("解:", [str(x) for x in seq + [a, a]], flush=True)
            raise SystemExit(0)
        k = (sem(np.array(o.grid)), o.pending)
        if k in seen:
            continue
        seen.add(k)
        q.append((seq + [a], ch, o))
    if expanded % 500 == 0:
        print(f"  扩展 {expanded} | 见过 {len(seen)} | 队列 {len(q)} | 最深 {deepest} | "
              f"{time.time()-t0:.0f}s", flush=True)

print(f"\n🚨**队列穷尽**: 扩展 {expanded}, 见过 {len(seen)} 个语义状态, 最深 {deepest}, "
      f"{time.time()-t0:.0f}s", flush=True)
print("=> 硬结论: 过关条件**不在 (画布 x 方块位置) 这个空间里**, 必有第三个维度。", flush=True)
