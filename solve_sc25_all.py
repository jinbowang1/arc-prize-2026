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
import heapq
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


def semantic_fp(game: Game, obs: Obs, acts: list[Action]):
    """语义指纹 = (画布内容, 构型指纹) —— 不是全屏。

    🚨L3 上一跑用**全屏** g.tobytes() 当指纹, 扩展 18000 节点、队列 23439、
    **最深只到 6 层**(人类 32 步)就打转。全屏里混着计数器、轨道渲染细节这些
    与决策无关的东西, 同一个语义状态被拆成无数份, 去重形同虚设。
    L1 破关时用的正是语义指纹(九宫格 + 上半屏), 33384 节点就搜到了。

    这里用 canvas 那套因果分解: **答案区(画布) + 掩码后的区外(构型)**,
    答案区靠 propose_prompt_answer 自动认(双粒度连通之后能提出整块九宫格),
    构型掩码靠 _config_mask 自动掩掉计数器(含"连着提交几次才现形"的那种)。
    """
    from harness.canvas import _config_fp, _config_mask, _region, classify
    from harness.hypo import propose_prompt_answer
    from harness.model import collect_states
    from harness.percept import mutable_over_states
    from harness.probe import run_probe
    sp = action_space(list(obs.actions))
    sc = analyze(obs.grid)
    clicks = [a for a in acts if a.id >= 6]
    rep = run_probe(game, obs, sp["kind"], sp["keys"], clicks)
    states = collect_states(game, obs, acts, 5)
    mut = mutable_over_states([lambda a, c=c: np.array(c.effect(a).grid) for c, _ in states],
                              [np.array(o.grid) for _, o in states], acts) & rep.mask
    props = propose_prompt_answer(np.array(obs.grid), mut, sc.bg)
    pick = st = None
    best = 0
    for h in props:
        t = classify(game, obs, acts, h.a)
        subs = [repr(a) for a in t.submitters]
        nc = sum(1 for r in subs if r.startswith("A6"))
        if subs and nc in (0, len(subs)) and len(subs) > best:
            pick, st, best = h, t, len(subs)
    if pick is None:
        print("  ⚠️认不出答案区, 退回全屏指纹", flush=True)
        return (lambda g: g.tobytes()), None, None, None, None
    BOX = pick.a
    mask = _config_mask(game, obs, st, BOX, rep.mask)

    # 🚨构型不能取"全屏减掩码" —— 掩码只掩掉计数器那几十格, 剩下 4059 格几乎还是
    # 全屏, 语义指纹就退化成全屏指纹(L3 实测: 扩展 2000/最深 4, 与全屏版一模一样)。
    # **构型的实质是"调整动作能改的那块"**(sc25 = 轨道上那个方块的位置/朝向),
    # 只占屏幕一小块。取调整动作可改格的包围盒, 状态空间才真正压下来。
    g0 = np.array(obs.grid)
    adj_mut = np.zeros_like(mask)
    for a in st.adjusters:
        o = game.effect(a)
        if not o.dead:
            adj_mut |= (np.array(o.grid) != g0)
    adj_mut &= mask
    r0, r1, c0, c1 = BOX
    adj_mut[r0:r1 + 1, c0:c1 + 1] = False        # 画布不算构型
    cells = np.argwhere(adj_mut)
    if len(cells) == 0:
        print(f"  语义指纹: 画布 {BOX} + 构型(全屏掩码 {int(mask.sum())} 格)", flush=True)
        return ((lambda g: (_region(g, BOX).tobytes(), _config_fp(g, BOX, mask))),
                BOX, _region(mut, BOX), None, sc.bg)
    CFG = (int(cells[:, 0].min()), int(cells[:, 0].max()),
           int(cells[:, 1].min()), int(cells[:, 1].max()))
    ch, cw = CFG[1] - CFG[0] + 1, CFG[3] - CFG[2] + 1
    print(f"  语义指纹: 画布 {BOX}(提交 {len(st.submitters)}) + 构型 {CFG} "
          f"({ch}x{cw}={ch*cw} 格, 而全屏掩码是 {int(mask.sum())} 格)", flush=True)
    return (lambda g: (_region(g, BOX).tobytes(), _region(g, CFG).tobytes()),
            BOX, _region(mut, BOX), CFG, sc.bg)


def solve_level(game: Game, obs: Obs, lv: int):
    """启发式搜索。h = 画布上"还不是目标色"的可变格数。

    🚨为什么不是 BFS: L3 上三种指纹(全屏 4096 / 语义 4059 / 压缩构型 144)
    结果**一模一样** —— 扩展 2000、最深都只有 4 层。换方法而数字纹丝不动,
    说明**指纹不是瓶颈**, 盲搜没有方向才是: 分支 14、深度 4 已经 14^4=38416,
    而解在 15 步开外。

    h 从哪来: **L1/L2 的真解**。两关过关瞬间画布都是**全色 2**(见 results
    README 第六节), 这是地面真值不是猜的。这里把它当作**跨关迁移的假设**用 ——
    猜错了 h 会降不下去, 那本身就是可读信号(与"加算力 h 不动"同类)。
    ⚠️判据仍然只认 level 上升, h 只负责排序(模型当排序器和当预测器是两条及格线)。
    """
    acts = act_pool(game, obs)
    t0 = time.time()
    fp, BOX, inbox, CFG, BG = semantic_fp(game, obs, acts)

    from harness.canvas import _region as _rg
    TCOL = 2                          # 画布目标色: 从 L1/L2 真解归纳

    # 🚨第二项: **清空构型区**。同样来自真解 —— L1 过关前那一帧, 轨道
    # (19,22,23,42) 里**一个非底色格都没有**, 方块被那 8 次 A3 推出去了。
    # 所以过关不是"把方块推到某位置", 是"把它清掉"。
    # 只有画布那一项时, h 三步就降到 0 然后**失去方向**(L3 实测: h=0 之后
    # 扩展 4000 仍停在最深 8) —— 画布达标是必要不充分, 第二项补的正是那一半。
    # ⚠️底色要取**构型区内的众数**, 不能用全屏背景色 scene.bg。
    # 实测栽过: 轨道底色是 2, 而 scene.bg=5(画面背景), 于是"清空"被算成
    # "把轨道全变成 5" —— **永远达不到**, h2 恒在 144 附近, 把画布那一项
    # 完全淹没。L2 太简单(34 节点)没被影响, L3 会被带偏。
    cfg_base = None
    if CFG is not None:
        from collections import Counter
        sub = _rg(np.array(obs.grid), CFG)
        cfg_base = Counter(sub.flatten().tolist()).most_common(1)[0][0]
    def h_of(g):
        h = 0
        if BOX is not None and inbox is not None:
            h += int(((_rg(g, BOX) != TCOL) & inbox).sum())
        if CFG is not None and cfg_base is not None:
            h += int((_rg(g, CFG) != cfg_base).sum())
        return h
    h0 = h_of(np.array(obs.grid))
    print(f"  启发式: 画布可变格 {int(inbox.sum()) if inbox is not None else 0} 个"
          f" + 构型区 {CFG}(目标=清空到众数色 {cfg_base}) | 开局 h={h0}", flush=True)

    seen = {(fp(np.array(obs.grid)), obs.pending)}
    heap = [(h0, 0, [], game.fork(), obs)]
    cnt = 0
    best_h = h0
    expanded = deepest = 0
    while heap and expanded < PER_LEVEL_NODES and time.time() - t0 < PER_LEVEL_WALL:
        _, _, seq, node, ob = heapq.heappop(heap)
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
            g = np.array(o.grid)
            k = (fp(g), o.pending)
            if k in seen:
                continue
            seen.add(k)
            hv = h_of(g)
            if hv < best_h:
                best_h = hv
                print(f"    h 降到 {hv} (深度 {len(seq)+1}, 扩展 {expanded}, "
                      f"{time.time()-t0:.0f}s)", flush=True)
            cnt += 1
            heapq.heappush(heap, (hv + 0.05 * (len(seq) + 1), cnt, seq + [a], ch, o))
        if expanded % 2000 == 0:
            print(f"    扩展 {expanded} | 见过 {len(seen)} | 堆 {len(heap)} | "
                  f"最深 {deepest} | h 最好 {best_h} | {time.time()-t0:.0f}s", flush=True)
    tag = "堆穷尽(硬结论)" if not heap else "触上限"
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
