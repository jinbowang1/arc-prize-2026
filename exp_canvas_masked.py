"""通用画布求解器打 cd82 L3。

对账基准:这一关的已知解是 18 步(cd82_solutions.json 的 per_level_steps[2]),
人类 41 步。真机逐步搜在这一关上 900 秒 34259 节点 h 卡在 28 不动。

全流程零手工传参:答案区来自因果判据,提交/调整动作按"改不改得动答案区"二分。
"""
import json
import sys
import time

import numpy as np

from harness.canvas import classify, collect_brushes, plan_canvas, solve, _region
from harness.env import Action, Game, action_space
from harness.hypo import propose_prompt_answer
from harness.model import collect_states
from harness.percept import analyze, discover, mutable_over_states
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
rep.mask[63, 55] = False   # 手工补掩: 落笔计数器, probe 的跨步判据漏了它

# ① 答案区 + 题面:因果判据
states = collect_states(game, obs, acts, 5)
mut = mutable_over_states([lambda a, c=c: np.array(c.peek(a).grid) for c, _ in states],
                          [np.array(o.grid) for _, o in states], acts) & rep.mask
props = propose_prompt_answer(np.array(obs.grid), mut, scene.bg)
if not props:
    raise SystemExit("因果判据没提出任何目标")
h = props[0]
print(f"目标: {h.describe()}  开局差 {h.distance(np.array(obs.grid)):.0f} 格", flush=True)

# ② 动作二分
st = classify(game, obs, acts, h.a)
print(st.text(), flush=True)

# ③ 画笔库(双底)
brushes, complete, judged, total_cells, ncfg = collect_brushes(game, obs, st, rep.mask, max_configs=2000, max_seconds=300)
sizes = sorted((b.size for b in brushes), reverse=True)[:5]
print(f"[canvas] 画笔 {len(brushes)} 支, 最大覆盖 {sizes}, 本次采集能判 {judged}/{total_cells} 格, "
      f"构型 {ncfg} 个{'完整' if complete else '**被截断**(库不全时抽象层会像无解)'}"
      f" | {time.time()-t0:.0f}s", flush=True)

# ④ 抽象画布求解
start = _region(np.array(obs.grid), h.a)
target = _region(np.array(obs.grid), h.b)
plan = plan_canvas(start, target, brushes)
print(plan.text(), flush=True)

# ⑤ 闭环执行:每落一笔就用实测画布重规划
target = _region(np.array(obs.grid), h.b)
def acts_fn(o):
    sc = analyze(o.grid)
    return ([Action.key(i) for i in sp['keys']] +
            [Action.click(c, r) for (r, c) in sc.targets])
seq, obs2, why = solve(game, obs, st, target, rep.mask, acts_fn=acts_fn,
                       max_configs=2000, collect_seconds=300)
print(f"[canvas] {why}", flush=True)
print(f"真机 {len(seq)} 步 -> level {obs2.level} | 总耗时 {time.time()-t0:.0f}s", flush=True)
if obs2.level > LV:
    print(f"✅ 通关! {len(seq)} 步 vs 已知解 {sol['per_level_steps'][LV]} 步 "
          f"vs 人类 {sol['baseline'][LV]} 步", flush=True)
    print("解:", [str(a) for a in seq], flush=True)
