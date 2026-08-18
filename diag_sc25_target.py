"""sc25 L3: judged 81/169 的分母对不对? 目标配对可不可达?

可疑的数字巧合: 九宫格 13x13 = 169 格, 但里面只有 **9 个 3x3 格子 = 81 格**是
可变的, 其余 88 格是格子之间的分隔(永不变)。而 judged 正好停在 **81**。
若果真如此, 那 81/169 已经是**判满了**, min_ratio 拿 169 当分母是错的,
会一直触发本不必要的轨迹底重采。

第二问: 抽象层差 126 —— 差在哪? 如果差异落在**分隔/固定格**上, 那是
**目标配对本身不可达**(尺寸凑对了但结构不同), 再怎么采集也解不出。
"""
from __future__ import annotations

import json

import numpy as np

from harness.canvas import _region, classify
from harness.env import Action, Game, action_space
from harness.hypo import propose_prompt_answer
from harness.model import collect_states
from harness.percept import analyze, mutable_over_states
from harness.probe import run_probe
from harness.run import _parse

PAL = " .:-=+*#%@$&XO<>"

sol = json.load(open("sc25_solutions.json"))
game, obs = Game.make("sc25")
for t in sol["seq"]:
    obs = game.act(_parse(t))
sp = action_space(list(obs.actions))
scene = analyze(obs.grid)
clicks = [Action.click(c, r) for (r, c) in scene.targets]
acts = [Action.key(i) for i in sp["keys"]] + clicks
game.detect_lag(acts)
rep = run_probe(game, obs, sp["kind"], sp["keys"], clicks)
states = collect_states(game, obs, acts, 5)
mut = mutable_over_states([lambda a, c=c: np.array(c.effect(a).grid) for c, _ in states],
                          [np.array(o.grid) for _, o in states], acts) & rep.mask
props = propose_prompt_answer(np.array(obs.grid), mut, scene.bg)

pick = st = None
best = 0
for h in props:
    t = classify(game, obs, acts, h.a)
    subs = [repr(a) for a in t.submitters]
    nc = sum(1 for r in subs if r.startswith("A6"))
    if subs and nc in (0, len(subs)) and len(subs) > best:
        pick, st, best = h, t, len(subs)

g0 = np.array(obs.grid)
A, B = pick.a, pick.b
ans, tgt = _region(g0, A), _region(g0, B)
inbox = _region(mut, A)
n_mut = int(inbox.sum())
print(f"答案区 {A} = {ans.shape[0]}x{ans.shape[1]} = {ans.size} 格", flush=True)
print(f"  区内**可变**格 {n_mut} 个 | 上一跑判得动 81", flush=True)
print(f"  => judged 的分母用了整块 {ans.size}; 真实覆盖率应是 81/{n_mut}"
      f" = {81 / max(1, n_mut):.0%}", flush=True)

print(f"\n答案区内容            题面 {B} 内容", flush=True)
for r in range(ans.shape[0]):
    a = "".join(PAL[v % 16] for v in ans[r])
    b = "".join(PAL[v % 16] for v in tgt[r])
    mark = "" if np.array_equal(ans[r], tgt[r]) else "  <>"
    print(f"  {a}    {b}{mark}", flush=True)

d = (ans != tgt)
print(f"\n答案区 vs 题面: 差 {int(d.sum())}/{ans.size} 格", flush=True)
print(f"  落在**可变格**上: {int((d & inbox).sum())} 格(这些能改)", flush=True)
print(f"  落在**固定格**上: {int((d & ~inbox).sum())} 格 <- **永远改不掉**", flush=True)
if int((d & ~inbox).sum()) > 0:
    print("\n🚨目标配对**不可达**: 差异有一部分落在改不动的格子上,"
          "\n   再怎么采集也解不出。尺寸凑对了不等于结构对得上。", flush=True)
