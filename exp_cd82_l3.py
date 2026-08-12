"""cd82 L3 决定性实验:目标对了之后,加算力值不值。

判据划清楚:h 从 100 降到 28(深度 39)说明**梯度指向目标**,这种情况下
加算力是对的。反面是 cd82 L4 那次 —— 换搜索策略而结论数字纹丝不动,
那是表征缺自由度,加多少 beam 都没用。
"""
import json
import numpy as np
from harness.env import Action, Game, action_space
from harness.hypo import propose_prompt_answer
from harness.model import collect_states
from harness.percept import analyze, discover, mutable_over_states
from harness.probe import probe_counters, probe_volatile, run_probe
from harness.run import _parse
from harness.search import best_first

sol = json.load(open("cd82_solutions.json"))
game, obs = Game.make("cd82")
for a in [_parse(s) for s in sol["seq"]][:11]:
    obs = game.act(a)
print("到 L3, level =", obs.level, flush=True)

sp = action_space(list(obs.actions))
scene = analyze(obs.grid)
clicks = [Action.click(c, r) for (r, c) in scene.targets]
acts = [Action.key(i) for i in sp["keys"]] + clicks
rep = run_probe(game, obs, sp["kind"], sp["keys"], clicks)
states = collect_states(game, obs, acts, 5)
mut = mutable_over_states([lambda a, c=c: np.array(c.peek(a).grid) for c, _ in states],
                          [np.array(o.grid) for _, o in states], acts) & rep.mask
h = propose_prompt_answer(np.array(obs.grid), mut, scene.bg)[0]
print("目标:", h.describe(), "开局 h =", h.distance(np.array(obs.grid)), flush=True)

res = best_first(game, obs, sp["keys"], 6, lambda n, o: h.distance(np.array(o.grid)),
                 rep.mask, max_depth=60, max_nodes=400000, max_seconds=900)
print(res.text(), flush=True)
if res.solved:
    print("解:", [str(a) for a in res.seq], flush=True)
