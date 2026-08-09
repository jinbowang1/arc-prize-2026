"""L5: 9 击基础 + 未覆盖格 (22,24) 的两种态, clone 验证。"""
import json
import numpy as np
import arc_agi
from arcengine import GameAction, ActionInput
from solve_ft09_all import clone, raw

arc = arc_agi.Arcade()
env = arc.make("ft09")
f = env.reset()
sols = json.load(open("ft09_solutions.json"))
for seq in sols["seqs"]:
    for (x, y) in seq:
        f = env.step(GameAction.ACTION6, {"x": x, "y": y})
game = env._game
level = f.levels_completed + 1

base = [(32, 6), (16, 22), (32, 22), (48, 22), (16, 38), (32, 38), (48, 38), (16, 54), (32, 54)]
for extra in ([], [(24, 22)]):
    clicks = base + extra
    ch = clone(game)
    win = False
    for (x, y) in clicks:
        fr = raw(ch, 6, {"x": x, "y": y})
        if fr.levels_completed >= level:
            win = True; break
    print(f"{len(clicks)}击 (extra={extra}): {'WIN' if win else '未过'}")
    if win:
        for (x, y) in clicks:
            f = env.step(GameAction.ACTION6, {"x": x, "y": y})
        print(f"真机 levels={f.levels_completed}")
        sols["seqs"].append(clicks)
        json.dump(sols, open("ft09_solutions.json", "w"))
        print("已存")
        break
