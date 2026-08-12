"""r11l L2 的结构诊断:动作之间是覆盖、交换,还是累积?

这个问题决定状态该怎么表示,而状态表示决定搜索规模:

  - **覆盖**(a 之后按 b,等于只按 b): 动作分成若干"槽",状态 = 每槽最后选了
    什么。状态数 = 各槽大小之积,不是 38^深度。
  - **交换**(ab 与 ba 同): 状态 = 已按动作的集合,不是序列。
  - **累积且有序**: 状态就是序列,只能硬搜。

不试图通关。L7 那次的教训:一个只问机制的诊断跑,比多搜十层都值钱。
"""
import numpy as np

from harness.env import Action, Game, action_space
from harness.percept import analyze
from harness.probe import probe_counters, probe_volatile
from harness.search import bfs_level_up


def fp(o, mask):
    return (np.array(o.grid) * mask).tobytes()


game, obs = Game.make("r11l")
sp = action_space(list(obs.actions))

# 先过 L1
scene = analyze(obs.grid)
clicks = [Action.click(c, r) for (r, c) in scene.targets]
mask = probe_volatile(game, obs, [], clicks) & probe_counters(game, obs, clicks)
res = bfs_level_up(game, obs, [], 6, mask, max_depth=20, max_nodes=20000, max_seconds=90)
print("L1:", res.text(), flush=True)
for a in res.seq:
    obs = game.act(a)

# L2 现场
scene = analyze(obs.grid)
clicks = [Action.click(c, r) for (r, c) in scene.targets]
mask = probe_volatile(game, obs, [], clicks) & probe_counters(game, obs, clicks)
base_fp = fp(obs, mask)

live = []
for a in clicks:
    o = game.peek(a)
    if not o.dead and fp(o, mask) != base_fp:
        live.append(a)
print(f"L2 有效动作 {len(live)} / 候选 {len(clicks)}", flush=True)

# 单动作结果表
single = {}
for a in live:
    single[repr(a)] = fp(game.peek(a), mask)
print(f"单动作产生 {len(set(single.values()))} 个不同状态(有效动作 {len(live)} 个)"
      f" —— 相同说明这些动作等效", flush=True)

# 两两关系: 在 base 上走 a 再走 b
n = min(12, len(live))
probe = live[:n]
override = commute = accumulate = 0
examples = {"override": [], "commute": [], "accumulate": []}
for a in probe:
    ca = game.fork()
    ca.act(a)
    for b in probe:
        if a == b:
            continue
        ab = fp(ca.fork().act(b), mask)
        cb = game.fork()
        cb.act(b)
        ba = fp(cb.fork().act(a), mask)
        b_only = single[repr(b)]
        if ab == b_only:
            override += 1
            if len(examples["override"]) < 3:
                examples["override"].append(f"{a}然后{b} == 只按{b}")
        elif ab == ba:
            commute += 1
            if len(examples["commute"]) < 3:
                examples["commute"].append(f"{a}{b} == {b}{a}")
        else:
            accumulate += 1
            if len(examples["accumulate"]) < 3:
                examples["accumulate"].append(f"{a}{b} != {b}{a} 且 != 只按{b}")

tot = override + commute + accumulate
print(f"\n两两关系({n} 个动作, {tot} 对):")
print(f"  覆盖(后盖前) {override} 对 {override/tot:.0%}  {examples['override']}")
print(f"  交换(序无关) {commute} 对 {commute/tot:.0%}  {examples['commute']}")
print(f"  累积(序有关) {accumulate} 对 {accumulate/tot:.0%}  {examples['accumulate']}")

# 两步能到多少个不同状态: 真实分支因子
seen = set()
for a in probe:
    ca = game.fork()
    ca.act(a)
    for b in probe:
        seen.add(fp(ca.fork().act(b), mask))
print(f"\n{n}x{n} 两步序列 -> {len(seen)} 个不同状态 "
      f"(若纯覆盖应约 {n} 个, 若纯交换应约 {n*(n+1)//2} 个, 若全累积应 {n*n} 个)")
