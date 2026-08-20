"""cd82 终验+计分: 全新环境重放全部解, 官方 scorecard 对账。

cd82 的解是键盘动作与点击混排的扁平序列("A3" / "A6(48,4)"),
按 per_level_steps 切回逐关, 每关末尾核对关卡数确实 +1。
"""
import json, re
import arc_agi
from arcengine import GameAction

STEP = re.compile(r"^A(\d)(?:\((\d+),(\d+)\))?$")

sols = json.load(open("cd82_solutions.json"))
seq, per_level = sols["seq"], sols["per_level_steps"]
assert len(seq) == sum(per_level) == sols["total"], "解的长度与分关步数对不上"

arc = arc_agi.Arcade()
env = arc.make("cd82")
f = env.reset()
cur = 0
for li, n in enumerate(per_level, 1):
    before = f.levels_completed
    for tok in seq[cur:cur + n]:
        m = STEP.match(tok)
        assert m, f"无法解析的动作 {tok!r}"
        act = getattr(GameAction, f"ACTION{m.group(1)}")
        data = {"x": int(m.group(2)), "y": int(m.group(3))} if m.group(2) else None
        f = env.step(act, data) if data else env.step(act)
    cur += n
    assert f.levels_completed == before + 1, f"L{li} 重放断裂!"
print(f"终验: {f.levels_completed}/{f.win_levels} state={f.state.name} 共{cur}步")

d = json.loads(arc.get_scorecard().model_dump_json())
run = max(d["environments"][0]["runs"], key=lambda r: r["levels_completed"])
base = run["level_baseline_actions"]
print(f"{'关':<4}{'我方':>6}{'人类':>6}{'得分':>9}")
for i, (act, bl, sc) in enumerate(zip(run["level_actions"], base, run["level_scores"]), 1):
    if act:
        print(f"L{i:<3}{act:>6}{bl:>6}{sc:>8.1f}%")
print(f"总动作 {run['actions']} (人类合计 {sum(base[:run['levels_completed']])}), 游戏得分 {run['score']:.2f}")
