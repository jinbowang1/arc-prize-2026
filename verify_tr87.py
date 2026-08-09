"""tr87 终验: 全新环境从头重放全部解, 打印每关步数与最终 scorecard。"""
import json
import arc_agi
from arcengine import GameAction

ACTS = {i: getattr(GameAction, f"ACTION{i}") for i in range(1, 8)}
sols = json.load(open("tr87_solutions.json"))
arc = arc_agi.Arcade()
env = arc.make("tr87")
f = env.reset()
total = 0
for li, seq in enumerate(sols["seqs"]):
    before = f.levels_completed
    for a in seq:
        f = env.step(ACTS[a])
    total += len(seq)
    print(f"L{li+1}: {len(seq)}步 -> levels={f.levels_completed} state={f.state.name}")
    assert f.levels_completed == before + 1, "重放断裂!"
print(f"\n总步数 {total}, 最终 state={f.state.name}")
sc = env.scorecard() if hasattr(env, "scorecard") else None
try:
    card = arc.scorecard_manager.get_scorecard() if hasattr(arc, "scorecard_manager") else None
except Exception:
    card = None
for attr in ("score", "level_scores", "level_actions", "level_baseline_actions"):
    for obj in (f, env, getattr(env, "_game", None)):
        if obj is not None and hasattr(obj, attr):
            print(f"{attr}: {getattr(obj, attr)}")
            break
