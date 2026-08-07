"""全程终验+计分: 全新环境重放完整序列, 用官方 scorecard 对账 RHAE。"""
import json, os
import arc_agi
from arcengine import GameAction

A = {1: GameAction.ACTION1, 2: GameAction.ACTION2, 3: GameAction.ACTION3, 4: GameAction.ACTION4}

# solutions.json 已含全部已通关卡的解(逐关合并), 不要再追加分关文件, 否则重复计数
seq = json.load(open("solutions.json"))["seq"]

arc = arc_agi.Arcade()
env = arc.make("ls20")
f = env.reset()
for a in seq:
    f = env.step(A[a])
print(f"终验: 通关 {f.levels_completed}/{f.win_levels}, 状态 {f.state.name}, 共 {len(seq)} 步")

d = json.loads(arc.get_scorecard().model_dump_json())
run = max(d["environments"][0]["runs"], key=lambda r: r["levels_completed"])
base = run["level_baseline_actions"]
print(f"{'关':<4}{'我方':>6}{'人类':>6}{'得分':>9}")
for i, (act, bl, sc) in enumerate(zip(run["level_actions"], base, run["level_scores"]), 1):
    if act:
        print(f"L{i:<3}{act:>6}{bl:>6}{sc:>8.1f}%")
print(f"总动作 {run['actions']} (人类基准合计 {sum(base[:run['levels_completed']])}), "
      f"游戏得分 {run['score']:.2f}")
