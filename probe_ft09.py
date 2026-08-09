"""ft09 初探: 全帧 + click 语义(坐标格式/点击块的反馈)。"""
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
CH = ".123456789ABCDEF"
print("=== 全帧 ===")
for r in range(64):
    print(f"{r:>3} " + "".join(CH[v] for v in g0[r]))

print("\n=== 点击试探(clone 上) ===")
for (y, x, why) in [(4, 6, "左上9色块"), (12, 14, "第二行图案块"), (4, 40, "右侧9块"), (0, 0, "背景角落")]:
    ch = clone(game)
    fr = raw(ch, 6, {"x": x, "y": y})
    if not fr.frame:
        print(f"({x},{y}) {why}: 无帧"); continue
    g = np.array(fr.frame[-1])
    d = np.argwhere(g != g0)
    rows = sorted({int(r) for r, _ in d})
    print(f"({x},{y}) {why}: 变动{len(d)}px 行{rows[:8]} 帧数{len(fr.frame)} state={fr.state.name} 关卡={fr.levels_completed}")
