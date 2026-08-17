"""诊断: 构型转移图上, 那 32 条"调整"边有多少条是真的?

上一个诊断(diag_skeleton.py)否掉了"构型转移依赖画布", 骨架 ② 成立。但它顺手
交出一条线索: 32 个 adjusters 在短序列上只造出 **3 个不同构型**, 20 组比对里
13 组走完压根没挪动构型。

嫌疑落在 classify 上, 它只做了一件事:
    改得动答案区的 -> submitters, **剩下的一律 -> adjusters**
从没问过"这个动作到底改不改构型"。而且只在开局那一个状态上问了一次。
于是 noop 被无条件当成图上的边, 名义出度 32, 真实出度未知。

把二分改成三分, 在多个构型上问:
    改构型          -> 真调整(图上的真边)
    只改画布不改构型 -> **本该是提交**, 开局那次恰好没改到答案区而漏判
    两者都不改       -> noop, 图上的虚边

如果虚边占大头, 那么 08-15 "从当前构型穷尽 56 个 / 从 anchor 终点穷尽 190 个"
这个逻辑上不可能的实测就有了不需要"画布依赖"的解释, 而 _ActionPool 那套动态
重算是在给一张本来就错的图打补丁。
"""
from __future__ import annotations

import json
import time
from collections import Counter

import numpy as np

from harness.canvas import _config_fp, _config_mask, _region, classify
from harness.env import Action, Game, Obs, action_space
from harness.hypo import propose_prompt_answer
from harness.model import collect_states
from harness.percept import analyze, mutable_over_states
from harness.probe import run_probe
from harness.run import _parse

GID, LV = "cd82", 2
SPREAD = 8          # 在多少个互不相同的构型上问

sol = json.load(open(f"{GID}_solutions.json"))
game, obs = Game.make(GID)
for a in [_parse(s) for s in sol["seq"]][:sum(sol["per_level_steps"][:LV])]:
    obs = game.act(a)

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
cmask = _config_mask(game, obs, st, BOX, rep.mask)
print(f"答案区 {BOX} | 名义: 提交 {len(st.submitters)} 调整 {len(st.adjusters)}", flush=True)


def fp(o: Obs) -> bytes:
    return _config_fp(np.array(o.grid), BOX, cmask)


def canvas(o: Obs) -> np.ndarray:
    return _region(np.array(o.grid), BOX)


# ── 采一批互不相同的构型当提问现场(只在开局问是这个项目栽过七次的老毛病) ──
sites: list[tuple[Game, Obs]] = [(game.fork(), obs)]
seen_cfg = {fp(obs)}
frontier = [(game.fork(), obs)]
while frontier and len(sites) < SPREAD:
    nd, ob = frontier.pop(0)
    for a in st.adjusters:
        if len(sites) >= SPREAD:
            break
        ch = nd.fork()
        o2 = ch.act(a)
        if o2.dead or o2.level != ob.level:
            continue
        f = fp(o2)
        if f in seen_cfg:
            continue
        seen_cfg.add(f)
        sites.append((ch, o2))
        frontier.append((ch.fork(), o2))
print(f"采到 {len(sites)} 个互不相同的构型当现场"
      f"{'  ⚠️不足 %d, 构型空间本身可能就很小' % SPREAD if len(sites) < SPREAD else ''}",
      flush=True)

# ── 逐个动作、逐个现场问三分类 ──
cfg_hit = Counter()      # 该动作在多少个现场改动了构型
canvas_hit = Counter()   # 该动作在多少个现场改动了画布
dead_hit = Counter()
outdeg: list[int] = []   # 每个现场的真实出度

for nd, o in sites:
    base_cfg, base_canvas = fp(o), canvas(o)
    deg = 0
    for a in st.adjusters:
        ch = nd.fork()
        o2 = ch.act(a)
        if o2.dead or o2.level != o.level:
            dead_hit[repr(a)] += 1
            continue
        if fp(o2) != base_cfg:
            cfg_hit[repr(a)] += 1
            deg += 1
        if not np.array_equal(canvas(o2), base_canvas):
            canvas_hit[repr(a)] += 1
    outdeg.append(deg)

real = [repr(a) for a in st.adjusters if cfg_hit[repr(a)]]
mislabeled = [repr(a) for a in st.adjusters
              if not cfg_hit[repr(a)] and canvas_hit[repr(a)]]
noop = [repr(a) for a in st.adjusters
        if not cfg_hit[repr(a)] and not canvas_hit[repr(a)]]

print(f"\n[三分类 · 在 {len(sites)} 个构型上问过]", flush=True)
print(f"  真调整(改构型, 图上真边)        {len(real):3d} 个", flush=True)
print(f"  🚨本该是提交(只改画布, 被漏判)  {len(mislabeled):3d} 个  {mislabeled[:6]}", flush=True)
print(f"  noop(什么都不改, 图上虚边)      {len(noop):3d} 个  {noop[:6]}", flush=True)

print(f"\n[真实出度] 名义 {len(st.adjusters)} | 各现场真实出度 {outdeg}", flush=True)
if outdeg:
    print(f"  中位 {sorted(outdeg)[len(outdeg)//2]} | 最小 {min(outdeg)} | 最大 {max(outdeg)}",
          flush=True)

# 真边也要看它是不是处处有效 —— 只在个别构型上有效的边, 静态图会当成处处能走
if real:
    print("\n[真边的有效范围] 该动作在几个现场改得动构型:", flush=True)
    for r in real:
        n = cfg_hit[r]
        flag = "" if n == len(sites) else "  ⚠️只在部分构型上有效"
        print(f"  {r:<22} {n}/{len(sites)}{flag}", flush=True)

virtual = len(mislabeled) + len(noop)
print(f"\n结论: {len(st.adjusters)} 条名义边里 {virtual} 条是假的 "
      f"({virtual / max(1, len(st.adjusters)):.0%})", flush=True)
# 🚨判据必须挂在"搜索真正付出的代价"上, 也就是**分支因子**, 不能拿动作条数比。
# 17 条真边 vs 15 条虚边看着"真边是多数", 而真实出度 14 vs 名义 32 是虚高 2.3 倍
# —— 后者才是 BFS 每层多扩出多少废节点。第一版判词就写成了前者。
med = sorted(outdeg)[len(outdeg) // 2] if outdeg else 0
ratio = len(st.adjusters) / med if med else float("inf")
print(f"  分支因子: 真实 {med} vs 名义 {len(st.adjusters)} = **虚高 {ratio:.1f} 倍**", flush=True)
# 🚨这个数字是真的, 但**它不是瓶颈, 别拿它去剪枝**。08-17 实测: 照着剔掉这
# 15 条边, 可达构型 1074 -> 56, 抽象层从"解出 4 笔"退成"解不出(差 14 格)",
# 采集 143s -> 10s 快得好看而已。
#   · BFS 自己会跳过 noop 边(构型指纹不变), 一条 noop 只值一次 fork = 常数;
#   · 剔错一条真边 = 可达集塌缩 = 漏解。
# 拿指数换常数, 方向是反的。这里的产出应该喂给**感知层**(scene.targets 把
# 控件之间的缝也当成了可点目标), 而不是喂给搜索当剪枝。
print("  ⚠️虚高是真的, 但**不要据此剪枝** —— 实测剔完可达构型塌 95%, 端到端更差。"
      "\n     这条线索的正确去处是感知层: 那些 noop 点击都落在控件之间的缝上。", flush=True)
# 边的可用性本身随构型变(A2 只在 2/8 现场有效) —— 这跟"依赖画布"是两件事,
# 骨架 ② 仍然成立。但静态边表会把它当成处处能走。
# ── 顺手当 prune_noops 的回归: 它剔出来的必须与上面三分类算出的**逐字相同** ──
from harness.canvas import prune_noops  # noqa: E402

st2 = classify(game, obs, acts, BOX)
ndrop, nkeep, nsites = prune_noops(game, obs, st2, BOX, cmask, apply=True)
got = sorted(repr(a) for a in st2.noops)
want = sorted(noop)
print(f"\n[prune_noops 回归] 剔 {ndrop} 留 {nkeep} (在 {nsites} 个构型上问)", flush=True)
if got == want:
    print(f"  ✅ 与三分类的 noop 清单逐字相同 ({len(got)} 个)", flush=True)
else:
    print(f"  ❌ 清单不一致\n     只在 prune 里: {sorted(set(got) - set(want))}"
          f"\n     只在诊断里:   {sorted(set(want) - set(got))}", flush=True)
if sorted(repr(a) for a in st2.adjusters) != sorted(real):
    print(f"  ❌ 留下的边与三分类的真边清单不一致", flush=True)

part = [r for r in real if cfg_hit[r] != len(sites)]
if part:
    print(f"  ⚠️{len(part)}/{len(real)} 条真边**只在部分构型上有效**"
          f"(最窄的 {min(cfg_hit[r] for r in part)}/{len(sites)})。"
          "\n     静态边表按'处处能走'来规划, 执行期就可能报走不到。", flush=True)
    # ⚠️别把这条当成"56 vs 190"那个矛盾的解释 —— 未证实, 而且前提已经动摇:
    # 08-17 基线实测(exp_canvas_openloop, 不剔边)可达构型 **1074 完整**,
    # 跟当时记的"从当前构型穷尽 56 个"对不上。到底是那次 BFS 另有预算/起点,
    # 还是记录本身有误, 得单独查 —— 别拿一句听着顺的解释把它盖过去。
print(f"总耗时 {time.time()-t0:.0f}s", flush=True)
