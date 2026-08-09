"""tr87 终验+计分: 官方 scorecard 对账。"""
import json
import arc_agi
from arcengine import GameAction

ACTS = {i: getattr(GameAction, f"ACTION{i}") for i in range(1, 8)}
sols = json.load(open("tr87_solutions.json"))
arc = arc_agi.Arcade()
env = arc.make("tr87")
f = env.reset()
for seq in sols["seqs"]:
    for a in seq:
        f = env.step(ACTS[a])
print(f"终验: {f.levels_completed}/{f.win_levels} state={f.state.name}")
d = json.loads(arc.get_scorecard().model_dump_json())
run = max(d["environments"][0]["runs"], key=lambda r: r["levels_completed"])
base = run["level_baseline_actions"]
print(f"{'关':<4}{'我方':>6}{'人类':>6}{'得分':>9}")
for i, (act, bl, sc) in enumerate(zip(run["level_actions"], base, run["level_scores"]), 1):
    if act:
        print(f"L{i:<3}{act:>6}{bl:>6}{sc:>8.1f}%")
print(f"总动作 {run['actions']} (人类合计 {sum(base[:run['levels_completed']])}), 游戏得分 {run['score']:.2f}")
