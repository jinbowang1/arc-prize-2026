"""L3 重跑: 全动作集 + 三值画布指纹。

上一版穷尽(55188 状态无解)**结论作废**, 两个原因:
 ① **动作集被筛窄**: 按"开局无效"剔掉 18 个点击, 其中 5 个
    (A6(11,55)/A6(12,55)/A6(15,51)/A6(15,54)/A6(15,57)) 是**只在画布涂过后
    才激活的按钮** —— 点它们把九宫格中间列(行49-61 列29-31)清空成 0。
    "在少数状态上验证的无效不等于处处无效", cd82 上栽过同款。
 ② **画布不是二值**: 之前靠"连点同一格"测出 0<->14 / 2<->14 就断定二值,
    但这些按钮能直接设 0 -> 画布是**三值(0/2/14)**, 空间 2^9 -> 3^9(涨 38 倍)。

所以这次: 动作 = 4 按键 + **全部 28 个点击**; 画布指纹取**实际颜色**不做二值化。
带补步查 level。设节点上限, 空间可能很大。
"""
from __future__ import annotations

import json
import time
from collections import deque

import numpy as np

from harness.env import Action, Game, action_space
from harness.percept import analyze
from harness.run import _parse

MAX_NODES = 250000
WALL = 2700.0

sol = json.load(open("sc25_solutions.json"))
game, obs = Game.make("sc25")
for t in sol["seq"]:
    obs = game.act(_parse(t))
L0 = obs.level
g0 = np.array(obs.grid)
sp = action_space(list(obs.actions))
sc = analyze(obs.grid)
ACTS = [Action.key(k) for k in sp["keys"]] + [Action.click(c, r) for (r, c) in sc.targets]
game.detect_lag(ACTS)
R, C = [49, 54, 59], [24, 29, 34]
print(f"L{L0+1}: 动作 {len(ACTS)} 个(按键 {len(sp['keys'])} + 点击 {len(sc.targets)}, **不筛**)",
      flush=True)

def sem(g):
    """画布 9 格的**实际颜色**(三值) + 方块位置"""
    cells = tuple(int(g[R[i] + 1, C[j] + 1]) for i in range(3) for j in range(3))
    sub = g[10:34, 26:46]
    m = ((sub == 9) | (sub == 10))
    c = np.argwhere(m)
    pos = (round(float(c[:, 0].mean()) + 10, 1),
           round(float(c[:, 1].mean()) + 26, 1)) if len(c) else None
    return (cells, pos)

t0 = time.time()
seen = {(sem(g0), obs.pending)}
q = deque([([], game.fork(), obs)])
expanded = deepest = 0
while q and expanded < MAX_NODES and time.time() - t0 < WALL:
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
            json.dump({"game": "sc25", "level": 3, "seq": [str(x) for x in seq + [a]]},
                      open("sc25_l3_solution.json", "w"), ensure_ascii=False, indent=1)
            raise SystemExit(0)
        p = ch.fork().act(a)
        if not p.dead and p.level > L0:
            print(f"🏆 过关(补步)! {len(seq)+2} 步 | 扩展 {expanded}", flush=True)
            print("解:", [str(x) for x in seq + [a, a]], flush=True)
            json.dump({"game": "sc25", "level": 3, "seq": [str(x) for x in seq + [a, a]]},
                      open("sc25_l3_solution.json", "w"), ensure_ascii=False, indent=1)
            raise SystemExit(0)
        k = (sem(np.array(o.grid)), o.pending)
        if k in seen:
            continue
        seen.add(k)
        q.append((seq + [a], ch, o))
    if expanded % 1000 == 0:
        print(f"  扩展 {expanded} | 见过 {len(seen)} | 队列 {len(q)} | 最深 {deepest} | "
              f"{time.time()-t0:.0f}s", flush=True)

tag = "队列穷尽" if not q else ("触节点上限" if expanded >= MAX_NODES else "触墙钟")
print(f"\n未解出({tag}): 扩展 {expanded}, 见过 {len(seen)}, 最深 {deepest}, "
      f"{time.time()-t0:.0f}s", flush=True)
