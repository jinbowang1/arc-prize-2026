"""sc25 L1: 启发式搜索(best-first), h = 九宫格与显示器图案的差异格数。

盲搜已经证明不行(sc25_l1_semantic_bfs.log): 12000 节点 / 17527 状态 / 最深仅 10 层。
分支 13 的 BFS 进不到解的深度, 所以需要 h 引导。

已确定的语义(diag 实测):
    点九宫格某格 = 该格 **2 <-> 14 切换**(不是单向环)
    画面**丢掉序列的第一个动作**: 点 N 次只结算 N-1 次
      · 点 1/2/3/4 次 -> 格(0,0) = 2/14/2/14
      · 垫任意个 A1 再点两次 -> 总是回到 2(结算 2 次 = 切换两次)
      · A1 打头会让相位翻转(它顶掉被丢的那个位置)
    A1-A4 只动轨道方块, **完全不影响九宫格**
    ⚠️残留: 空格(色0) 按此模型点两次该到 14, 实测是 2 —— 语义没完全闭合,
      所以**判据只认 level > 0**, h 只用来排序(模型当排序器和当预测器是两条
      不同的及格线)。

目标猜测(未证实): 九宫格色 14 的位置 == 显示器色 15 的位置, 即
    (0,1) (1,0) (1,2) (2,1) 为 14, 其余为 2
h = 与该图案不一致的格数。猜错了 h 就是噪声, 那就会表现为"h 降不下去",
和加算力不动一样是个可读的信号。
"""
from __future__ import annotations

import heapq
import json
import time

import numpy as np

from harness.env import Action, Game

R, C = [49, 54, 59], [24, 29, 34]
TRACK = (19, 22, 23, 42)
CLICKS = [Action.click(C[j] + 1, R[i] + 1) for i in range(3) for j in range(3)]
KEYS = [Action.key(k) for k in (1, 2, 3, 4)]
ACTS = CLICKS + KEYS

WANT = {(0, 1), (1, 0), (1, 2), (2, 1)}      # 显示器色15 的位置
TARGET = tuple(14 if (i, j) in WANT else 2 for i in range(3) for j in range(3))


def cells(g):
    return tuple(int(g[r + 1, c + 1]) for r in R for c in C)


def sem(g):
    r0, r1, c0, c1 = TRACK
    return (cells(g), g[r0:r1 + 1, c0:c1 + 1].tobytes())


def h(g):
    return sum(1 for a, b in zip(cells(g), TARGET) if a != b)


def solve(max_nodes=20000, max_seconds=1500.0):
    game, obs = Game.make("sc25")
    g0 = np.array(obs.grid)
    print(f"开局九宫格 {cells(g0)}  目标 {TARGET}  h={h(g0)}", flush=True)
    t0 = time.time()
    seen = {(sem(g0), 0)}
    cnt = 0
    heap = [(h(g0), 0, [], game.fork(), obs)]
    best = h(g0)
    expanded = 0
    while heap and expanded < max_nodes and time.time() - t0 < max_seconds:
        _, _, seq, node, ob = heapq.heappop(heap)
        expanded += 1
        for a in ACTS:
            child = node.fork()
            o = child.act(a)
            if o.dead:
                continue
            if o.level > 0:
                print(f"🏆 过关! {len(seq)+1} 步 (人类 36) | 扩展 {expanded} | "
                      f"{time.time()-t0:.0f}s", flush=True)
                return seq + [a]
            g = np.array(o.grid)
            key = (sem(g), o.pending)
            if key in seen:
                continue
            seen.add(key)
            hv = h(g)
            if hv < best:
                best = hv
                print(f"  h 降到 {hv} (深度 {len(seq)+1}, 扩展 {expanded}, "
                      f"{time.time()-t0:.0f}s) 九宫格 {cells(g)}", flush=True)
            cnt += 1
            heapq.heappush(heap, (hv + 0.1 * (len(seq) + 1), cnt, seq + [a], child, o))
        if expanded % 500 == 0:
            print(f"  扩展 {expanded} | 见过 {len(seen)} | 堆 {len(heap)} | "
                  f"h 最好 {best} | {time.time()-t0:.0f}s", flush=True)
    print(f"未解出: 扩展 {expanded}, 见过 {len(seen)}, h 最好 {best}, "
          f"{time.time()-t0:.0f}s", flush=True)
    return None


sol = solve()
if sol:
    json.dump({"game": "sc25", "level": 1, "seq": [str(a) for a in sol]},
              open("sc25_l1_solution.json", "w"), ensure_ascii=False, indent=1)
    g2, o2 = Game.make("sc25")
    for a in sol:
        o2 = g2.act(a)
    print(f"[复核] 全新环境重放 {len(sol)} 步: level={o2.level} state={o2.state} "
          f"{'✅' if o2.level > 0 else '❌'}", flush=True)
