"""验证: 修掉计数器之后, 开环执行抽象层的完整方案能不能通。

诊断结论(diag_last12.py): 卡住的 12 格是一个必须**先涂**的块 —— 有 5 支笔颜色
全对, 但落下去会盖坏已涂好的部分(cd82 是后涂盖先涂)。抽象层开局解出的 4 笔
方案本来含着正确顺序, 是**闭环的贪心把顺序拆散了**。

而当初改成闭环的理由是"开环第 4 笔找不到摆法" —— 那次失败的真因是计数器
(63,55) 漏进构型指纹导致采集退化, 今天已修。前提没了, 结论也不成立。

所以这里直接开环: 开局采一次库 -> 抽象层解完整方案 -> execute 逐笔翻译回真机。
⚠️execute 内部的去重指纹也必须用修正后的构型掩码, 否则计数器会让去重失效。
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from harness.canvas import (_config_mask, _region, classify, collect_brushes,
                            execute_cfg, plan_canvas)
from harness.env import Action, Game, action_space
from harness.hypo import propose_prompt_answer
from harness.model import collect_states
from harness.percept import analyze, mutable_over_states
from harness.probe import run_probe
from harness.run import _parse

GID = sys.argv[1] if len(sys.argv) > 1 else "cd82"
LV = int(sys.argv[2]) if len(sys.argv) > 2 else 2

sol = json.load(open(f"{GID}_solutions.json"))
game, obs = Game.make(GID)
for a in [_parse(s) for s in sol["seq"]][:sum(sol["per_level_steps"][:LV])]:
    obs = game.act(a)
print(f"到 L{LV+1}, level={obs.level}, 已知解 {sol['per_level_steps'][LV]} 步, "
      f"人类 {sol['baseline'][LV]} 步", flush=True)

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
print(f"目标: {h.describe()}  开局差 {h.distance(np.array(obs.grid)):.0f} 格", flush=True)

st = classify(game, obs, acts, BOX)
print(st.text(), flush=True)

# 构型掩码: 提交动作改得动的答案区外格子一律不算构型(计数器就是这么划出去的)
cmask = _config_mask(game, obs, st, BOX, rep.mask)
extra = np.argwhere(rep.mask & ~cmask)
print(f"[掩码] probe 掩 {int((~rep.mask).sum())} 格, 因果判据再掩 "
      f"{len(extra)} 格: {extra[:8].tolist()}", flush=True)

brushes, complete, judged, total, ncfg = collect_brushes(
    game, obs, st, rep.mask, max_configs=2000, max_seconds=300)
print(f"[canvas] 画笔 {len(brushes)} 支, 判 {judged}/{total} 格, 构型 {ncfg} "
      f"{'完整' if complete else '**截断**'} | {time.time()-t0:.0f}s", flush=True)

start = _region(np.array(obs.grid), BOX)
target = _region(np.array(obs.grid), h.b)

# 证据账要分两个口径报, 混成一个会自欺:
#  - 跨笔并集: 每格是否**存在某支笔**说得清 -> 好看, 但不决定抽象层能不能收尾
#  - 单笔盲区: **收尾那一笔**自己还有多少格不知道 -> 这个才是地板
#    (apply 会把一支笔的 unknown 区整片涂成 UNKNOWN, 哪怕那些格之前已经涂对)
unk_all = np.ones_like(start, dtype=bool)
for b in brushes:
    unk_all &= (b.unknown if b.unknown is not None else np.zeros_like(unk_all))
unk_any = np.zeros_like(start, dtype=bool)
for b in brushes:
    if b.unknown is not None:
        unk_any |= b.unknown
sizes = sorted(int(b.unknown.sum()) if b.unknown is not None else 0 for b in brushes)
print(f"[证据] 跨笔并集判不了 {int(unk_all.sum())} 格 | "
      f"至少一支笔判不了 {int(unk_any.sum())} 格", flush=True)
print(f"       单笔盲区: 最小 {sizes[0]} 格, 中位 {sizes[len(sizes)//2]} 格, "
      f"最大 {sizes[-1]} 格 <- **收尾要用盲区为 0 的笔**", flush=True)
print(f"       盲区为 0 的笔: {sum(1 for s in sizes if s == 0)} 支", flush=True)
if unk_any.any():
    print("[至少一支笔判不了的格] o=全部笔都有证据 .=有笔没证据", flush=True)
    for r in range(unk_any.shape[0]):
        print("    " + " ".join("." if unk_any[r, c] else "o"
                                for c in range(unk_any.shape[1])), flush=True)

plan = plan_canvas(start, target, brushes)
print(plan.text(), flush=True)
if not plan.found:
    raise SystemExit(f"抽象层没解出: 最好差 {plan.best_gap}")

for i, cum in enumerate(plan.cumulative):
    print(f"  第 {i+1} 笔之后应有差异 {int((cum != target).sum())} 格", flush=True)

# 开环执行。跑在克隆体上, 真机不动。
node = game.fork()
seq, o2, why = execute_cfg(node, obs, st, plan, cmask)
print(f"[execute] {why}", flush=True)
print(f"真机 {len(seq)} 步 -> level {o2.level} | 总耗时 {time.time()-t0:.0f}s", flush=True)
if o2.level > LV:
    print(f"✅ 通关! {len(seq)} 步 vs 已知解 {sol['per_level_steps'][LV]} 步 "
          f"vs 人类 {sol['baseline'][LV]} 步", flush=True)
    print("解:", [str(a) for a in seq], flush=True)
