"""ft09: 36 块全扫分类反馈。"""
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

cols = [6, 14, 22, 40, 48, 56]
rows = [4, 12, 20, 38, 46, 54]
for y in rows:
    line = []
    for x in cols:
        ch = clone(game)
        fr = raw(ch, 6, {"x": x, "y": y})
        gl = np.array(fr.frame[-1])
        dl = np.argwhere(gl != g0)
        interim = sum(len(np.argwhere(np.array(fr.frame[k]) != (np.array(fr.frame[k-1]) if k else g0))) for k in range(len(fr.frame)))
        tag = "静默"
        if fr.levels_completed > 0:
            tag = "过关!"
        elif len(dl):
            rws = sorted({int(r) for r, _ in dl})
            tag = f"终变{len(dl)}px@{rws[0]}-{rws[-1]}"
        elif interim:
            tag = "闪烁拒绝"
        line.append(f"({x:>2},{y:>2}){tag}")
    print(" ".join(line), flush=True)
