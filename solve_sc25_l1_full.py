"""sc25 L1: 点击 + 按键 组合 BFS, 带"补步查 level"。过夜跑。

两个**队列穷尽**的硬结论(不是超时), 把方向夹死了:
    只用九宫格点击 -> 992 状态穷尽, 无解   (sc25_l1_clicks_only_exhausted.log)
    只用按键 A1-A4 ->  23 状态穷尽, 无解
=> 过关必须两者配合。组合空间约 23 x 992 ~ 2 万量级, BFS 穷尽得起
   (上一跑 12000 节点时队列还剩 5527, 远没搜完)。

两个修正:
  ① **补步查 level**: 画面滞后一拍已确认, 过关信号没理由不滞后。每扩展一步后
     多 fork 一次走同样动作, 看 level 是否上升 —— 只查当前 o.level 会漏掉
     "最后一步已达标但还没显示"的解。这可能正是之前一直漏解的原因。
  ② **指纹含两个子系统**: 九宫格颜色 + 上半屏(插槽 列12-16 + 轨道 列17-42)。
     少了任一个都会把有效状态去重掉("指纹要正好等于语义状态", 今晚撞第五次)。

已实测否掉的目标猜测(别再重走):
    九宫格全 2 / 全 14 / 只点显示器色15 标记的四格(72 种顺序x前置组合全试过)
    -> "九宫格色14 位置 == 显示器色15 位置"这个目标是**错的**
所以这里**不猜目标**, 判据只认 level 上升。
"""
from __future__ import annotations

import json
import time
from collections import deque

import numpy as np

from harness.env import Action, Game

R, C = [49, 54, 59], [24, 29, 34]
CLICKS = [Action.click(C[j] + 1, R[i] + 1) for i in range(3) for j in range(3)]
KEYS = [Action.key(k) for k in (1, 2, 3, 4)]
ACTS = CLICKS + KEYS
MAX_NODES = 120000
WALL = 5 * 3600.0


def sem(g: np.ndarray):
    return (tuple(int(g[r + 1, c + 1]) for r in R for c in C),   # 九宫格
            g[15:25, 10:45].tobytes())                            # 插槽 + 轨道


def main():
    game, obs = Game.make("sc25")
    g0 = np.array(obs.grid)
    t0 = time.time()
    seen = {(sem(g0), 0)}
    q = deque([([], game.fork(), obs)])
    expanded = 0
    deepest = 0
    while q and expanded < MAX_NODES and time.time() - t0 < WALL:
        seq, node, ob = q.popleft()
        expanded += 1
        deepest = max(deepest, len(seq))
        for a in ACTS:
            ch = node.fork()
            o = ch.act(a)
            if o.dead:
                continue
            if o.level > 0:
                return seq + [a], expanded, t0, "直接"
            # ① 补步查 level: 过关信号可能滞后一拍
            p = ch.fork().act(a)
            if not p.dead and p.level > 0:
                return seq + [a, a], expanded, t0, "补步"
            k = (sem(np.array(o.grid)), o.pending)
            if k in seen:
                continue
            seen.add(k)
            q.append((seq + [a], ch, o))
        if expanded % 500 == 0:
            print(f"  扩展 {expanded} | 见过 {len(seen)} | 队列 {len(q)} | "
                  f"最深 {deepest} | {time.time()-t0:.0f}s", flush=True)
    tag = "队列穷尽(硬结论: 这个状态空间里无解)" if not q else "触上限"
    print(f"未解出: 扩展 {expanded}, 见过 {len(seen)}, 队列剩 {len(q)}, "
          f"最深 {deepest}, {time.time()-t0:.0f}s —— {tag}", flush=True)
    return None, expanded, t0, tag


sol, expanded, t0, how = main()
if sol:
    print(f"🏆 过关! {len(sol)} 步 (人类 36) | {how}发现 | 扩展 {expanded} | "
          f"{time.time()-t0:.0f}s", flush=True)
    print("解:", [str(a) for a in sol], flush=True)
    json.dump({"game": "sc25", "level": 1, "seq": [str(a) for a in sol], "how": how},
              open("sc25_l1_solution.json", "w"), ensure_ascii=False, indent=1)
    # 🚨全新环境整条重放复核 —— 搜索进程内自报通关不算数(ls20 L6 踩过)
    g2, o2 = Game.make("sc25")
    for a in sol:
        o2 = g2.act(a)
    print(f"[复核] 全新环境重放 {len(sol)} 步: level={o2.level} state={o2.state} "
          f"{'✅ 通过' if o2.level > 0 else '❌ 不可复现'}", flush=True)
