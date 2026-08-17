"""sc25 L1: 语义状态 BFS —— 指纹只取九宫格, 判据用真实过关。

看懂的结构(render_sc25.log / diag_sc25_grid.py):
    九宫格 行49/54/59 x 列24/29/34, 每格 3x3 像素
    每格颜色环 0 -> 2 -> 14 (点击推进, 到 14 后不再变)
    显示器(行51-58 列12-19) 有个 3x3 图案: 色15 在 .X./X.X/.X.
    九宫格开局 色2 在 X.X/.X./X.X, 空(色0) 在互补位置
    A1-A4 只动轨道上那个双色块, **完全不影响九宫格**

已否掉的目标猜测(都实测过, 别再试):
    九宫格全 2      -> 不过关
    九宫格全 14     -> 不过关
    只把显示器标记的四格点到 14 -> 手工序列没摆成(滞后语义没推准)

所以这里**不猜目标**, 判据就用 `level > 0`。
关键是**指纹只取九宫格 9 格颜色 + pending**:
  - 全屏指纹会把轨道方块的无关变化算进去, 状态空间炸开(harness 裸跑就是这样,
    BFS 最深只到 5-6 层);
  - 九宫格只有 3^9 = 19683 种, 加 pending 也搜得完。
  这条就是"指纹要正好等于语义状态"的又一次应用。

⚠️pending 必须进指纹: sc25 的动作要下一次调用才结算, 同一画面下"缓冲里待结算
的是谁"不同就是不同状态(见 search.fingerprint 的注释)。
"""
from __future__ import annotations

import json
import time
from collections import deque

import numpy as np

from harness.env import Action, Game

R, C = [49, 54, 59], [24, 29, 34]
TRACK = (19, 22, 23, 42)      # 轨道: 那个双色块在这里移动/旋转/翻转
CLICKS = [Action.click(C[j] + 1, R[i] + 1) for i in range(3) for j in range(3)]
KEYS = [Action.key(k) for k in (1, 2, 3, 4)]


def cells(g: np.ndarray) -> tuple:
    return tuple(int(g[r + 1, c + 1]) for r in R for c in C)


def sem(g: np.ndarray):
    """语义状态 = 九宫格颜色 + 轨道内容。**两个子系统都要进指纹。**

    🚨只放九宫格是错的, 而且是我现犯的: A1-A4 只动方块、不动九宫格, 于是
    按键产生的状态全部撞上"已见过"被去重掉, 方块那一整个维度根本没被探索。
    第一版就这么跑了一轮 —— "指纹要正好等于语义状态"今晚第四次。

    (第一版的收获仍然有效: 只用九宫格点击时 BFS **队列穷尽** —— 992 个状态
     全搜完没有过关, 所以过关必须让方块参与。那是硬结论, 不是超时。)
    """
    r0, r1, c0, c1 = TRACK
    return (cells(g), g[r0:r1 + 1, c0:c1 + 1].tobytes())


def solve(acts: list[Action], max_nodes: int, label: str):
    game, obs = Game.make("sc25")
    g0 = np.array(obs.grid)
    print(f"\n=== {label}: 动作 {len(acts)} 个, 上限 {max_nodes} 节点 ===", flush=True)
    print(f"开局九宫格 {cells(g0)}", flush=True)
    t0 = time.time()
    start = (sem(g0), 0)
    seen = {start}
    q = deque([([], game.fork(), obs)])
    best = None
    expanded = 0
    while q and expanded < max_nodes:
        seq, node, ob = q.popleft()
        expanded += 1
        for a in acts:
            child = node.fork()
            o = child.act(a)
            if o.dead:
                continue
            if o.level > 0:
                dt = time.time() - t0
                print(f"🏆 过关! {len(seq)+1} 步 (人类 36) | 扩展 {expanded} 节点 {dt:.0f}s",
                      flush=True)
                return seq + [a]
            key = (sem(np.array(o.grid)), o.pending)
            if key in seen:
                continue
            seen.add(key)
            q.append((seq + [a], child, o))
        if expanded % 200 == 0:
            print(f"  扩展 {expanded} | 见过 {len(seen)} 状态 | 队列 {len(q)} | "
                  f"最深 {len(seq)+1} | {time.time()-t0:.0f}s", flush=True)
    print(f"  未解出: 扩展 {expanded}, 见过 {len(seen)} 状态, 队列剩 {len(q)}, "
          f"{time.time()-t0:.0f}s", flush=True)
    return None


# 先只用九宫格点击(分支 9); 不行再把按键加进来当"哑动作/提交"用
sol = solve(CLICKS + KEYS, 12000, "点击 + 按键(指纹含轨道)")

if sol:
    with open("sc25_l1_solution.json", "w") as f:
        json.dump({"game": "sc25", "level": 1, "seq": [str(a) for a in sol]},
                  f, ensure_ascii=False, indent=1)
    # 全新环境整条重放复核 —— 搜索进程内自报通关不算数
    g2, o2 = Game.make("sc25")
    for a in sol:
        o2 = g2.act(a)
    print(f"\n[复核] 全新环境重放 {len(sol)} 步: level={o2.level} state={o2.state} "
          f"{'✅' if o2.level > 0 else '❌'}", flush=True)
