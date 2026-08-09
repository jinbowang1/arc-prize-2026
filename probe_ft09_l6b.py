"""L6: 非常规点击位置探测(花纹/背景/块角/铆钉)。"""
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
g0 = np.array(f.frame[-1])

tests = [("花纹1", 14, 8), ("花纹2", 38, 24), ("花纹3", 22, 32), ("花纹4", 46, 48),
         ("背景", 2, 30), ("参考块", 6, 8), ("块(16,14)左上角", 12, 14), ("块(16,14)铆钉", 16, 14)]
for name, x, y in tests:
    ch = clone(game)
    fr = raw(ch, 6, {"x": x, "y": y})
    frames_diff = []
    prev = g0
    for fi, fim in enumerate(fr.frame):
        g = np.array(fim)
        d = np.argwhere(g[:63] != prev[:63])
        if len(d):
            rows = sorted({int(r) for r, _ in d})
            frames_diff.append(f"帧{fi}:{len(d)}px行{rows[0]}-{rows[-1]}")
        prev = g
    print(f"{name} click({x},{y}): {' | '.join(frames_diff) or '零反应'} 关卡={fr.levels_completed}")
