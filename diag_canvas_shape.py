"""诊断: 画笔的形状到底依赖什么?

canvas.py 现在把「笔的形状」和「走到它的路径」绑在同一个 Brush 里, 所以每落
一笔就整库重采 —— 而越到后面越造不出底(cd82 L3 实测 65 -> 30 -> 0 格),
库就瞎了, 最后卡在差 12 格。

如果形状**只是构型的函数、与画布无关**, 那形状就该在开局采一次(那时底最好造),
之后只重算路径。这个脚本验证的就是这一条, 判据是行为级的:

    拿开局采到的笔, 去预测「落了 N 笔之后」同一构型下提交的结果, 逐格比对。

不需要造底 —— 所以它能在采集判据失效的那个状态下照样问出机制。
这也是项目里反复应验的那条: 一个只回答「机制是什么」的诊断跑, 比多搜十层都值钱。

顺带验前提 B: 构型图与画布无关(两处 BFS 出的构型指纹集合是否同一个)。
"""
from __future__ import annotations

import json
import sys
import time
from collections import deque

import numpy as np

from harness.canvas import _config_fp, _region, classify, collect_brushes, plan_canvas
from harness.env import Action, Game, Obs, action_space
from harness.hypo import propose_prompt_answer
from harness.model import collect_states
from harness.percept import analyze, mutable_over_states
from harness.probe import run_probe
from harness.run import _parse

GID = sys.argv[1] if len(sys.argv) > 1 else "cd82"
LV = int(sys.argv[2]) if len(sys.argv) > 2 else 2
MAX_CFG = int(sys.argv[3]) if len(sys.argv) > 3 else 80

sol = json.load(open(f"{GID}_solutions.json"))
game, obs = Game.make(GID)
for a in [_parse(s) for s in sol["seq"]][:sum(sol["per_level_steps"][:LV])]:
    obs = game.act(a)
print(f"到 L{LV+1}, level={obs.level}", flush=True)

t0 = time.time()
sp = action_space(list(obs.actions))
scene = analyze(obs.grid)
clicks = [Action.click(c, r) for (r, c) in scene.targets]
acts = [Action.key(i) for i in sp["keys"]] + clicks
rep = run_probe(game, obs, sp["kind"], sp["keys"], clicks)

states = collect_states(game, obs, acts, 5)
mut = mutable_over_states([lambda a, c=c: np.array(c.peek(a).grid) for c, _ in states],
                          [np.array(o.grid) for _, o in states], acts) & rep.mask
h = propose_prompt_answer(np.array(obs.grid), mut, scene.bg)[0]
BOX = h.a
st = classify(game, obs, acts, BOX)
print(f"答案区 {BOX} | 提交 {len(st.submitters)} 调整 {len(st.adjusters)}", flush=True)


def build_config_map(g: Game, o: Obs, limit: int) -> dict[bytes, list[Action]]:
    """从当前状态 BFS 遍历构型, 返回 构型指纹 -> 到达它的调整序列。

    构型指纹已经把答案区挖掉了, 所以这张图理论上与画布无关 —— 这正是要验的。
    """
    start_fp = _config_fp(np.array(o.grid), BOX, rep.mask)
    out: dict[bytes, list[Action]] = {start_fp: []}
    q: deque[tuple[list[Action], Game, Obs]] = deque([([], g.fork(), o)])
    while q and len(out) < limit:
        seq, node, ob = q.popleft()
        for a in st.adjusters:
            child = node.fork()
            c = child.act(a)
            if c.dead or c.level != o.level:
                continue
            fp = _config_fp(np.array(c.grid), BOX, rep.mask)
            if fp in out:
                continue
            out[fp] = seq + [a]
            q.append((seq + [a], child, c))
    return out


# ---- 前提 A: 提交动作改不改构型 ----------------------------------------
# 🚨 必须在多个构型上问, 不能只在开局问一次。"采样只在一个状态上做" 在这个项目里
# 已经栽过四次(搜索候选表 / 槽搜索动作表 / 可变格 / 动作二分)。
cfg0 = build_config_map(game, obs, MAX_CFG)
print(f"[A] 开局构型图 {len(cfg0)} 个 | {time.time()-t0:.0f}s", flush=True)

bad_a = 0
tested_a = 0
for i, (fp, seq) in enumerate(list(cfg0.items())[:12]):
    node = game.fork()
    ok = True
    for a in seq:
        if node.act(a).dead:
            ok = False
            break
    if not ok:
        continue
    for sub in st.submitters:
        c = node.fork()
        o2 = c.act(sub)
        if o2.dead:
            continue
        tested_a += 1
        if _config_fp(np.array(o2.grid), BOX, rep.mask) != fp:
            bad_a += 1
print(f"[A] 提交动作不改构型: {tested_a - bad_a}/{tested_a} 次成立"
      f"{'  ✅' if bad_a == 0 else '  ❌ 骨架不成立!'}", flush=True)

# ---- 采开局画笔库 -------------------------------------------------------
brushes, complete, judged, total, ncfg = collect_brushes(
    game, obs, st, rep.mask, max_configs=MAX_CFG, max_seconds=120)
print(f"[B0] 开局画笔 {len(brushes)} 支, 能判 {judged}/{total} 格 | {time.time()-t0:.0f}s",
      flush=True)

def walk(g: Game, o: Obs, seq: list[Action]) -> tuple[Game, Obs] | tuple[None, None]:
    """在克隆体上走一条序列, 返回终点。中途死了就报 None。"""
    node = g.fork()
    cur = o
    for a in seq:
        cur = node.act(a)
        if cur.dead:
            return None, None
    return node, cur


# 按**构型指纹**索引形状: 走完 seq 到哪个构型, 那支笔的形状就挂在那个指纹上。
# 这一步就是「把形状层和路径层拆开」—— 现在的 Brush 把两者绑死了。
shape: dict[tuple[bytes, str], tuple[np.ndarray, np.ndarray]] = {}
for b in brushes:
    node, cur = walk(game, obs, b.seq)
    if node is None:
        continue
    fp = _config_fp(np.array(cur.grid), BOX, rep.mask)
    shape[(fp, str(b.submit))] = (b.covered, b.stroke)
print(f"[B0] 形状按构型索引: {len(shape)} 条 (构型 × 提交动作)", flush=True)

# ---- 落 N 笔, 走到「采集判据失效」的那个状态 ---------------------------
start = _region(np.array(obs.grid), BOX)
target = _region(np.array(obs.grid), h.b)
plan = plan_canvas(start, target, brushes)
print(f"[plan] {plan.text()}", flush=True)
if not plan.found:
    raise SystemExit("抽象层没解出方案, 诊断到此为止")

node = game.fork()
cur = obs
laid = 0
for b in plan.brushes[:3]:
    n2, c2 = walk(node, cur, b.seq)
    if n2 is None:
        break
    c3 = n2.act(b.submit)
    if c3.dead or c3.level != obs.level:
        break
    node, cur, laid = n2, c3, laid + 1
print(f"[落笔] 已落 {laid} 笔, level={cur.level}, "
      f"当前离目标差 {int((_region(np.array(cur.grid), BOX) != target).sum())} 格", flush=True)

# ---- 前提 B: 构型图与画布无关 ------------------------------------------
cfgN = build_config_map(node, cur, MAX_CFG)
both = set(cfg0) & set(cfgN)
print(f"[B] 落笔后构型图 {len(cfgN)} 个, 与开局重合 {len(both)} 个"
      f" (开局独有 {len(set(cfg0)-set(cfgN))}, 落笔后独有 {len(set(cfgN)-set(cfg0))})"
      f"{'  ✅ 同一张图' if len(both) == len(cfg0) == len(cfgN) else '  ⚠️ 不是同一张图'}",
      flush=True)

# ---- 前提 C(核心): 开局采的形状能不能预测后期的实测结果 ----------------
# 判据是行为级的: 对每个共有构型, 在后期状态下真机提交一次, 拿开局的
# (covered, stroke) 去预测, 逐格比对。**不需要造底**, 所以在采集判据失效的
# 状态下照样问得出来。
hit = miss = 0
per_brush = []
for fp in list(both)[:40]:
    n2, c2 = walk(node, cur, cfgN[fp])
    if n2 is None:
        continue
    before = _region(np.array(c2.grid), BOX)
    for sub in st.submitters:
        key = (fp, str(sub))
        if key not in shape:
            continue
        covered, stroke = shape[key]
        c3 = n2.fork().act(sub)
        if c3.dead or c3.level != obs.level:
            continue
        after = _region(np.array(c3.grid), BOX)
        pred = before.copy()
        pred[covered] = stroke[covered]
        agree = int((pred == after).sum())
        # 只在这支笔声称盖到的格上判 —— 没盖到的格预测=保持原样, 那是白送的
        on_cov = int((pred == after)[covered].sum()) if covered.any() else 0
        ncov = int(covered.sum())
        per_brush.append((ncov, on_cov, agree, after.size))
        hit += on_cov
        miss += ncov - on_cov

print(f"[C] 用开局的笔预测落 {laid} 笔之后的实测结果:", flush=True)
if per_brush:
    full = sum(1 for n, oc, ag, sz in per_brush if ag == sz)
    print(f"    比对 {len(per_brush)} 支笔 | 声称覆盖的格上命中 {hit}/{hit+miss} "
          f"= {100*hit/max(1,hit+miss):.1f}%", flush=True)
    print(f"    **整块答案区逐格全对的笔: {full}/{len(per_brush)}** "
          f"(整帧判据 —— 逐格准会被背景白送, 只有整块全对才算这支笔是真的)", flush=True)
    print(f"    结论: {'✅ 形状只依赖构型, 可以开局采一次复用' if full == len(per_brush) else '❌ 形状还依赖画布之外的东西, 破法不成立'}",
          flush=True)
else:
    print("    没有可比对的笔 —— 共有构型上一支开局笔都没挂上, 说明索引这一步就断了",
          flush=True)
print(f"总耗时 {time.time()-t0:.0f}s", flush=True)
