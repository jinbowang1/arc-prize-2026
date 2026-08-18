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
                BOX, _region(mut, BOX), None, None)
    CFG = (int(cells[:, 0].min()), int(cells[:, 0].max()),
           int(cells[:, 1].min()), int(cells[:, 1].max()))
    ch, cw = CFG[1] - CFG[0] + 1, CFG[3] - CFG[2] + 1
    print(f"  语义指纹: 画布 {BOX}(提交 {len(st.submitters)}) + 构型 {CFG} "
          f"({ch}x{cw}={ch*cw} 格, 而全屏掩码是 {int(mask.sum())} 格)", flush=True)
    return (lambda g: (_region(g, BOX).tobytes(), _region(g, CFG).tobytes()),
            BOX, _region(mut, BOX), CFG, adj_mut)


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
    fp, BOX, inbox, CFG, adj_mut = semantic_fp(game, obs, acts)
    r0, r1, c0, c1 = BOX if BOX is not None else (0, 0, 0, 0)

    from harness.canvas import _region as _rg
    TCOL = 2                          # 画布目标色: 从 L1/L2 真解归纳

    # 🚨第二项: **清空构型区**。同样来自真解 —— L1 过关前那一帧, 轨道
    # (19,22,23,42) 里**一个非底色格都没有**, 方块被那 8 次 A3 推出去了。
    # 所以过关不是"把方块推到某位置", 是"把它清掉"。
    # 只有画布那一项时, h 三步就降到 0 然后**失去方向**(L3 实测: h=0 之后
    # 扩展 4000 仍停在最深 8) —— 画布达标是必要不充分, 第二项补的正是那一半。
    # 🚨h2 = **把"能动的那块"送到"同色但不能动的那块"去**(通用的"送货"启发式)。
    #
    # 走到这一版之前错了两次, 都记在这里:
    #  ①"清空构型区"的目标色取了全屏背景 scene.bg=5, 而轨道底色是 2 ->
    #    目标**不可达**, h2 恒 ~144 把 h1 完全淹没(开局 h=162 里 128 是噪声)。
    #    ⚠️h 曲线照样在降、L2 照样通关, 光看数字发现不了; 是把中间量打出来
    #    跟渲染图对照才抓到的。
    #  ②改成"清空到众数色"后 h 能降到 0 了, 但 **L3 上 h=0 仍不过关**:
    #    方块离开构型区就算 0, 可它被推到哪儿并不管。L1 恰好是"推出去就过关",
    #    所以这条在 L1 上看着对 —— **单关验证过的目标不等于跨关成立**。
    #
    # L3 真面目是**推箱子**: 按键 A1/A2/A3/A4 = 上/下/左/右各移 4 格,
    # 方块(色 9/10, 中心 (23.5,36.5))要送进**同为色 9/10 的插槽**(行37-42 列22-26)。
    # 于是判据变成"可动块与固定同色块重合", 用中心的曼哈顿距离当 h。
    g1 = np.array(obs.grid)
    mov_mask = adj_mut.copy() if adj_mut is not None else None
    fixed_ctr = mov_colors = None
    if mov_mask is not None and mov_mask.any():
        mov_colors = {int(g1[r, c]) for r, c in np.argwhere(mov_mask)}
        # ⚠️要**剔掉构型区的众数色**(轨道/管道的底色)。方块移动时, 它经过的格子
        # 由底色变成方块色、离开的格子又变回底色 —— 底色因此也进了 adj_mut。
        # 不剔的话"可动块"里混着遍布整条管道的底色, 中心被拉到管道正中:
        # L3 实测算出 (27.0,33.2)(管道中心) 而不是插槽, 开局 h 只有 19,
        # 而方块到插槽的真实曼哈顿距离约 28。
        from collections import Counter as _C
        if CFG is not None:
            _sub = _rg(g1, CFG)
            _bg = _C(_sub.flatten().tolist()).most_common(1)[0][0]
            mov_colors.discard(int(_bg))
        same = np.zeros_like(mov_mask)
        for col in mov_colors:
            same |= (g1 == col)
        fixed = same & ~mov_mask
        fixed[r0:r1 + 1, c0:c1 + 1] = False        # 画布内的同色不算目的地
        fc = np.argwhere(fixed)
        if len(fc) >= 4:
            fixed_ctr = (float(fc[:, 0].mean()), float(fc[:, 1].mean()))

    def mov_center(g):
        if not mov_colors:
            return None
        m = np.zeros((64, 64), dtype=bool)
        for col in mov_colors:
            m |= (g == col)
        m[r0:r1 + 1, c0:c1 + 1] = False
        if fixed_ctr is not None:                  # 排除目的地本身
            fr, fc2 = fixed_ctr
            for rr in range(64):
                for cc in range(64):
                    if m[rr, cc] and abs(rr - fr) <= 3 and abs(cc - fc2) <= 3:
                        m[rr, cc] = False
        cells = np.argwhere(m)
        return (float(cells[:, 0].mean()), float(cells[:, 1].mean())) if len(cells) else None
    def h_of(g):
        h = 0
        if BOX is not None and inbox is not None:
            h += int(((_rg(g, BOX) != TCOL) & inbox).sum())
        if fixed_ctr is not None:
            mc = mov_center(g)
            # 可动块不见了(送到了/被消掉) -> 距离记 0
            h += 0 if mc is None else int(abs(mc[0] - fixed_ctr[0]) + abs(mc[1] - fixed_ctr[1]))
        return h
    h0 = h_of(np.array(obs.grid))
    print(f"  启发式: 画布可变格 {int(inbox.sum()) if inbox is not None else 0} 个"
          f" + 送货(可动色 {mov_colors} -> 固定同色中心 {fixed_ctr}) | 开局 h={h0}", flush=True)

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
