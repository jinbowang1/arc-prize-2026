"""L7 初探: 四个方向各走一步, 看格子/形状/颜色/能量怎么变, 以及画面差异在哪。"""
import copy
import numpy as np
from wm import Percept, energy, load_env, panel_color, shape_bits
from arcengine import GameAction, ActionInput

ACTS = {a: getattr(GameAction, f"ACTION{a}") for a in (1, 2, 3, 4)}


def raw(g, a): return g.perform_action(ActionInput(id=ACTS[a], data={}), raw=True)


def clone(g, clean):
    g._clean_levels = None; g2 = copy.deepcopy(g); g._clean_levels = clean
    g2._clean_levels = clean; return g2


game, f = load_env("solutions_l6.json")
clean = game._clean_levels
g0 = np.array(f.frame[-1])
p = Percept(g0)
print("起点", p.key(g0), shape_bits(g0), panel_color(g0), energy(g0))

for a in (1, 2, 3, 4):
    ch = clone(game, clean)
    fr = raw(ch, a)
    if not fr.frame:
        print(f"a={a}: 无帧"); continue
    g = np.array(fr.frame[-1])
    pp = Percept(g0)
    diff = np.argwhere(g != g0)
    rows = sorted({int(r) for r, _ in diff})
    print(f"a={a}: 格={pp.key(g)} 形状={shape_bits(g)} 色={panel_color(g)} "
          f"能量={energy(g)} 关={fr.levels_completed} 变动像素={len(diff)} 涉及行={rows[:12]}")

# 连走 8 步 ACTION2, 看能否移动/是否有网格步长
ch = clone(game, clean)
prev = g0
for i in range(1, 9):
    fr = raw(ch, 2)
    if not fr.frame:
        print(f"步{i}: 死亡"); break
    g = np.array(fr.frame[-1])
    pp = Percept(prev)
    print(f"步{i} a=2: 格={pp.key(g)} 形状={shape_bits(g)} 能量={energy(g)} 关={fr.levels_completed}")
    prev = g
