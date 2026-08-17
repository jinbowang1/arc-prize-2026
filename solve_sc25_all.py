"""sc25 逐关求解: 有效动作筛选 + 全屏语义指纹 + 补步查 level。过夜跑。

L1 已用同一套方法破了(13 步 vs 人类 36, 整条重放复核通过):
    解 = A6(25,50) A6(30,50) A6(25,55) A6(35,55) A3 A3 A3 A3 A6(30,60) A3 A3 A3 A3
    **靠"补步查 level"才认出来** —— 只查当前 o.level 会整个漏掉。

harness 裸跑同一关却过不了, 差在两处(attack_sc25_settled.log):
    分支因子 27 vs 13  —— scene.targets 给 23 个点击目标, 其中 13 个点了全屏一格不变
    预算      135s vs 1159s
所以这里两件事都做: **筛掉真 noop** + **给足预算**。

⚠️剔除的判据必须是"**在多个状态上都改不动全屏任何一格**"。
这跟 08-17 在 cd82 上失败的那次剔除不是一回事: 那次剔的是"改不动构型"的边,
而构型图稀疏、边的可用性随构型变, 只在 8 个相邻构型上采样就永久剔除, 结果
可达构型 1074 -> 56 塌了 95%。这里的判据强得多(全屏无变化), 且同样在多个
状态上验证。
"""
from __future__ import annotations

import json
import time
from collections import deque

import numpy as np

from harness.env import Action, Game, Obs, action_space
from harness.percept import analyze

GID = "sc25"
BASE = [36, 6, 32, 83, 143, 50]
PER_LEVEL_NODES = 120000
PER_LEVEL_WALL = 3600.0


def act_pool(game: Game, obs: Obs) -> list[Action]:
    """当前关的动作集 = 按键 + **在多个状态上验证过有效**的点击目标。"""
    sp = action_space(list(obs.actions))
    sc = analyze(obs.grid)
    keys = [Action.key(k) for k in sp["keys"]]
    clicks = [Action.click(c, r) for (r, c) in sc.targets]
    game.detect_lag(keys + clicks)

    # 多状态采样: 开局 + 走几步之后, 每个状态上都问一遍
    sites = [game.fork()]
    n = game.fork()
    for k in keys[:3]:
        if not n.act(k).dead:
            sites.append(n.fork())
    live = []
    for a in clicks:
        for st in sites:
            before = np.array(st._grid())
            o = st.effect(a)
            if not o.dead and not np.array_equal(np.array(o.grid), before):
                live.append(a)
                break
    print(f"  点击目标 {len(clicks)} 个 -> 有效 {len(live)} 个 | 按键 {len(keys)} | "
          f"lagged={game.lagged}", flush=True)
    return keys + live


def solve_level(game: Game, obs: Obs, lv: int):
    acts = act_pool(game, obs)
    t0 = time.time()
    fp = lambda g: g.tobytes()
    seen = {(fp(np.array(obs.grid)), obs.pending)}
    q = deque([([], game.fork(), obs)])
    expanded = deepest = 0
    while q and expanded < PER_LEVEL_NODES and time.time() - t0 < PER_LEVEL_WALL:
        seq, node, ob = q.popleft()
        expanded += 1
        deepest = max(deepest, len(seq))
        for a in acts:
            ch = node.fork()
            o = ch.act(a)
            if o.dead:
                continue
            if o.level > ob.level:
                return seq + [a], expanded, time.time() - t0, "直接"
            # 补步查 level: 过关信号滞后一拍(L1 就是这么找到的)
            if game.lagged:
                p = ch.fork().act(a)
                if not p.dead and p.level > ob.level:
                    return seq + [a, a], expanded, time.time() - t0, "补步"
            k = (fp(np.array(o.grid)), o.pending)
            if k in seen:
                continue
            seen.add(k)
            q.append((seq + [a], ch, o))
        if expanded % 2000 == 0:
            print(f"    扩展 {expanded} | 见过 {len(seen)} | 队列 {len(q)} | "
                  f"最深 {deepest} | {time.time()-t0:.0f}s", flush=True)
    tag = "队列穷尽(硬结论)" if not q else "触上限"
    return None, expanded, time.time() - t0, tag


game, obs = Game.make(GID)
full: list[Action] = []
per_level: list[int] = []

# L1 已破(13 步, solve_sc25_l1_full.py, 整条重放复核过) —— 直接重放, 不重搜。
# 重搜一遍要 ~1200s, 纯属浪费; 而且真机推进到 L2 本来就得走这些步。
import os
if os.path.exists("sc25_l1_solution.json"):
    from harness.run import _parse
    l1 = [_parse(t) for t in json.load(open("sc25_l1_solution.json"))["seq"]]
    for a in l1:
        obs = game.act(a)
    full, per_level = list(l1), [len(l1)]
    print(f"L1 用在案解重放 {len(l1)} 步 -> level={obs.level} (人类 {BASE[0]})", flush=True)

for lv in range(len(per_level), 6):
    print(f"\n=== L{lv+1} (人类 {BASE[lv]} 步) ===", flush=True)
    sol, expanded, dt, how = solve_level(game, obs, lv)
    if sol is None:
        print(f"  ✗ 未解出 ({how}, 扩展 {expanded}, {dt:.0f}s)", flush=True)
        break
    for a in sol:
        obs = game.act(a)
    full += sol
    per_level.append(len(sol))
    print(f"  🏆 L{lv+1} 通关 {len(sol)} 步 vs 人类 {BASE[lv]} | {how}发现 | "
          f"扩展 {expanded} | {dt:.0f}s | level={obs.level}", flush=True)
    json.dump({"game": GID, "seq": [str(a) for a in full],
               "per_level_steps": per_level, "baseline": BASE},
              open(f"{GID}_solutions.json", "w"), ensure_ascii=False, indent=1)

print(f"\n通关 {len(per_level)}/6 | AI {len(full)} 步 vs 人类 {sum(BASE[:len(per_level)])}",
      flush=True)
if full:
    # 🚨全新环境整条重放复核 —— 搜索进程内自报不算数
    g2, o2 = Game.make(GID)
    for a in full:
        o2 = g2.act(a)
    print(f"[复核] 重放 {len(full)} 步: level={o2.level} state={o2.state} "
          f"{'✅' if o2.level >= len(per_level) else '❌'}", flush=True)
