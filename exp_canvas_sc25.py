"""拿 canvas 试解 sc25 L3 —— 验证"构型 x 提交"这条路对不对。

顺序实验已经定性(results/README.md 第六节): sc25 的两个子系统不可分段,
同样的点击在不同方块状态下涂出的颜色不同 => **方块 = 画笔颜色选择器,
点九宫格 = 用当前颜色提交**。这正是 `harness/canvas.py` 专门为之写的结构。

而 BFS 这条路已经用数据否掉了: L3 扩展 18000 节点、队列 23439、**最深卡在 6 层**,
人类基准 32 步。加宽没有意义。

这一跑只回答一个问题: **canvas 在 sc25 L3 上认不认得出这个结构**。
    答案区认成什么(应当是九宫格)
    提交/调整怎么分(应当是 点击=提交 / 按键=调整)
    画笔库采得到几支、判得动几格
不接主控(那是更大的改动), 先验证方向。

⚠️前置: 重放 L1+L2 在案解共 18 步推进到 L3。
"""
from __future__ import annotations

import json
import time

import numpy as np

from harness.canvas import (Budget, _config_mask, _region, classify,
                            collect_brushes, plan_canvas, solve_committed)
from harness.env import Action, Game, action_space
from harness.hypo import propose_prompt_answer
from harness.model import collect_states
from harness.percept import analyze, click_targets, mutable_over_states
from harness.probe import run_probe
from harness.run import _parse

GID = "sc25"
sol = json.load(open(f"{GID}_solutions.json"))
seq = [_parse(t) for t in sol["seq"]]
game, obs = Game.make(GID)
for a in seq:
    obs = game.act(a)
print(f"重放在案解 {len(seq)} 步 -> level={obs.level} (现在是 L{obs.level+1})", flush=True)

t0 = time.time()
sp = action_space(list(obs.actions))
scene = analyze(obs.grid)
clicks = [Action.click(c, r) for (r, c) in scene.targets]
acts = [Action.key(i) for i in sp["keys"]] + clicks
game.detect_lag(acts)
print(f"动作 {len(acts)} 个 (按键 {len(sp['keys'])} + 点击 {len(clicks)}) "
      f"| lagged={game.lagged}", flush=True)

rep = run_probe(game, obs, sp["kind"], sp["keys"], clicks)
states = collect_states(game, obs, acts, 5)
# 🚨用 effect 而不是 peek: sc25 上单步 peek 看到的是上一个动作的效果
mut = mutable_over_states([lambda a, c=c: np.array(c.effect(a).grid) for c, _ in states],
                          [np.array(o.grid) for _, o in states], acts) & rep.mask
props = propose_prompt_answer(np.array(obs.grid), mut, scene.bg)
print(f"\n因果目标提议 {len(props)} 条, 前 3:", flush=True)
for h in props[:3]:
    print(f"  {h}", flush=True)

BOX = props[0].a
st = classify(game, obs, acts, BOX)
print(f"\n{st.text()}", flush=True)
print(f"  提交动作: {[repr(a) for a in st.submitters][:8]}", flush=True)
print(f"  调整动作: {[repr(a) for a in st.adjusters][:8]}", flush=True)

mask = _config_mask(game, obs, st, BOX, rep.mask)
print(f"  构型掩码有效 {int(mask.sum())} 格", flush=True)

print("\n[采画笔库]", flush=True)
brushes, complete, judged, total, ncfg = collect_brushes(
    game, obs, st, rep.mask, max_configs=800,
    budget=Budget(max_expansions=6000, wall_seconds=1200.0), acts_fn=lambda o: acts)
print(f"  画笔 {len(brushes)} 支 | 判 {judged}/{total} 格 | 构型 {ncfg}"
      f"{'完整' if complete else '(截断)'} | {time.time()-t0:.0f}s", flush=True)

if brushes:
    canvas = _region(np.array(obs.grid), BOX)
    tgt = _region(np.array(obs.grid), props[0].b)
    plan = plan_canvas(canvas, tgt, brushes)
    print(f"  {plan.text()}", flush=True)
print(f"\n总耗时 {time.time()-t0:.0f}s", flush=True)
