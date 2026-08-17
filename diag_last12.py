"""诊断: 卡住的那 12 格到底是什么?

采集判据从 0/100 修到 30/100 之后, 差异**仍然是 12 格**, 一格没动。按这个项目
反复应验的判据 —— **换方法而数字纹丝不动, 八成是表征缺了自由度, 不是判据不够强**
—— 该停下来问机制, 不该接着调判据。

所以直接问三件事:
  1. 那 12 格在答案区的哪里, 现在什么色, 目标要什么色
  2. 画笔库里有没有任何一支笔盖得到这些格
  3. 盖得到的笔里, 有没有哪支的颜色是对的
如果答案是"一支都没有", 那缺的是笔(采集/构型覆盖不够);
如果"有笔但颜色不对", 那缺的是换色那一维。
"""
from __future__ import annotations

import json
import time

import numpy as np

from harness.canvas import (CanvasSetup, _region, classify, collect_brushes,
                            solve)
from harness.env import Action, Game, action_space
from harness.hypo import propose_prompt_answer
from harness.model import collect_states
from harness.percept import analyze, mutable_over_states
from harness.probe import run_probe
from harness.run import _parse

GID, LV = "cd82", 2
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
target = _region(np.array(obs.grid), h.b)
st = classify(game, obs, acts, BOX)


def acts_fn(o):
    sc = analyze(o.grid)
    return ([Action.key(i) for i in sp["keys"]] +
            [Action.click(c, r) for (r, c) in sc.targets])


seq3, _o, why = solve(game, obs, st, target, rep.mask, max_strokes=3,
                      acts_fn=acts_fn, max_configs=400)
node, cur = game.fork(), obs
for a in seq3:
    cur = node.act(a)
canvas = _region(np.array(cur.grid), BOX)
gap = (canvas != target)
print(f"落 3 笔 {len(seq3)} 步, 差 {int(gap.sum())} 格 | {time.time()-t0:.0f}s", flush=True)

print("\n[1] 差的是哪些格 (行,列 局部坐标): 现在色 -> 目标色", flush=True)
cells = np.argwhere(gap)
for (r, c) in cells:
    print(f"    ({r},{c}): {canvas[r,c]} -> {target[r,c]}", flush=True)
rows = sorted(set(int(r) for r, _ in cells))
cols = sorted(set(int(c) for _, c in cells))
print(f"    集中在 行 {rows} 列 {cols}", flush=True)
print(f"    目标色分布 {dict(zip(*np.unique(target[gap], return_counts=True)))}", flush=True)
print("\n[答案区现状]", flush=True)
for r in range(canvas.shape[0]):
    line = "".join(f"{canvas[r,c]:2d}" if not gap[r, c] else f" \033[31m{target[r,c]}\033[0m"
                   for c in range(canvas.shape[1]))
    print(f"    {line}", flush=True)
print("    (红色 = 差异格, 显示的是**目标**要的色)", flush=True)

# 这个状态下重新做动作二分, 再采一次库
fresh = classify(node, cur, acts_fn(cur), BOX)
subs = {repr(a): a for a in st.submitters}
subs.update({repr(a): a for a in fresh.submitters})
adjs = {repr(a): a for a in st.adjusters}
adjs.update({repr(a): a for a in fresh.adjusters})
for k in subs:
    adjs.pop(k, None)
st3 = CanvasSetup(answer_box=BOX, submitters=list(subs.values()),
                  adjusters=list(adjs.values()))
brushes, complete, judged, total, ncfg = collect_brushes(
    node, cur, st3, rep.mask, max_configs=400)
print(f"\n[2] 此状态画笔 {len(brushes)} 支, 判 {judged}/{total} 格, 构型 {ncfg} "
      f"{'完整' if complete else '**截断**'} | {time.time()-t0:.0f}s", flush=True)

reach = [b for b in brushes if (b.covered & gap).any()]
print(f"    盖得到这 12 格里任意一格的笔: {len(reach)} 支", flush=True)
if reach:
    best = None
    for b in reach:
        hit = int((b.covered & gap).sum())
        right = int(((b.stroke == target) & b.covered & gap).sum())
        after = int((b.apply(canvas) != target).sum())
        if best is None or after < best[0]:
            best = (after, b, hit, right)
    print(f"    最好的一支: 盖到 {best[2]} 个差异格, 其中颜色对的 {best[3]} 个, "
          f"落下去之后差 {best[0]} 格(现在 {int(gap.sum())})", flush=True)
    ok = [b for b in reach if ((b.stroke == target) | ~(b.covered & gap))[b.covered & gap].all()
          and (b.covered & gap).any()]
    print(f"    **盖到的差异格颜色全对的笔: {len(ok)} 支**", flush=True)
    print(f"    结论: {'缺的是颜色维度 —— 够得着但涂错色' if not ok else '有对的笔, 那问题在规划不在采集'}",
          flush=True)
else:
    print("    **一支都够不着** -> 缺的是笔本身, 即构型覆盖不够 / 采集判不动",
          flush=True)

# 这 12 格历史上被盖到过吗? 用开局的库对照
b0, _c, j0, t_, n0 = collect_brushes(game, obs, st, rep.mask,
                                     max_configs=400)
reach0 = [b for b in b0 if (b.covered & gap).any()]
print(f"\n[3] 对照: **开局**的库里够得着这 12 格的笔 {len(reach0)}/{len(b0)} 支 "
      f"(判 {j0}/{t_} 格)", flush=True)
if reach0:
    ok0 = [b for b in reach0
           if ((b.stroke == target) | ~(b.covered & gap))[b.covered & gap].all()]
    print(f"    其中颜色全对的 {len(ok0)} 支 -> "
          f"{'开局有、落笔后没了 = 采集退化' if ok0 else '开局也没有颜色对的'}", flush=True)
print(f"\n总耗时 {time.time()-t0:.0f}s", flush=True)
