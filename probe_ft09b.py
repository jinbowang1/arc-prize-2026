"""ft09: 逐帧 diff + reset 基线帧数确认。"""
import copy
import numpy as np
import arc_agi
from arcengine import GameAction, ActionInput

def raw(g, a, data=None):
    return g.perform_action(ActionInput(id=getattr(GameAction, f"ACTION{a}"), data=data or {}), raw=True)

def clone(g):
    clean = getattr(g, "_clean_levels", None)
    g._clean_levels = None; g2 = copy.deepcopy(g)
    g._clean_levels = clean; g2._clean_levels = clean
    return g2

arc = arc_agi.Arcade()
env = arc.make("ft09")
f = env.reset()
game = env._game
g0 = np.array(f.frame[-1])
print(f"reset 帧数: {len(f.frame)}")

# 块中心坐标: 6x6块, 左组列 4/12/20, 右组列 38/46/54; 行 2/10/18/36/44/52 -> 中心 +3
cols = [6, 14, 22, 40, 48, 56]
rows = [4, 12, 20, 38, 46, 54]
tests = [(rows[i], cols[j]) for i in (1, 4) for j in (1, 4)] + [(34, 46), (46, 46)]
for (y, x) in tests:
    ch = clone(game)
    fr = raw(ch, 6, {"x": x, "y": y})
    diffs = []
    prev = g0
    for fi, fim in enumerate(fr.frame):
        g = np.array(fim)
        d = np.argwhere(g != prev)
        if len(d):
            rows_ = sorted({int(r) for r, _ in d})
            diffs.append(f"帧{fi}:{len(d)}px行{rows_[:4]}")
        prev = g
    print(f"click({x:>2},{y:>2}): {' | '.join(diffs) or '全程零变动'} 关卡={fr.levels_completed}")
