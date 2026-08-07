"""验证 tr87 的结构假设: 上方=映射表, 中间=题面, 下方=可编辑答案区。

假设:
  - 行 4-28 的 6 对图案框 = 映射表(A色符号 ↔ 7色符号)
  - 行 40-46 = 7 个 A色符号(题面, 只读)
  - 行 51-57 = 7 个 7色符号(答案区, 可编辑)
  - ACTION3/4 移动光标, ACTION1/2 切换当前位的符号
验证方法: 连按 ACTION1 看当前位是否在有限个符号间循环, 循环长度应等于映射表的对数。
"""
import copy
import numpy as np
import arc_agi
from arcengine import GameAction, ActionInput

ACTS = {i: getattr(GameAction, f"ACTION{i}") for i in range(1, 8)}


def raw(g, a): return g.perform_action(ActionInput(id=ACTS[a], data={}), raw=True)


def clone(g):
    clean = getattr(g, "_clean_levels", None)
    g._clean_levels = None; g2 = copy.deepcopy(g)
    g._clean_levels = clean; g2._clean_levels = clean
    return g2


def cell(g, r0, c0, h=5, w=5):
    """取一个符号的 5x5 位掩码(非边框色即为亮)。"""
    w_ = g[r0:r0 + h, c0:c0 + w]
    return tuple(map(tuple, (w_ == 5).astype(int)))


def show(m):
    return "/".join("".join("X" if v else "." for v in row) for row in m)


arc = arc_agi.Arcade()
env = arc.make("tr87")
f = env.reset()
game = env._game
g0 = np.array(f.frame[-1])

# 框的精确列: A框 12-18 / 7框 22-28 / A框2 35-41 / 7框2 45-51 -> 内部各去掉一圈
print("=== 映射表(行4-28 的 6 对图案) ===")
for i, r in enumerate((5, 14, 23)):
    for j, (ca, cb) in enumerate(((13, 23), (36, 46))):
        a, b = cell(g0, r, ca), cell(g0, r, cb)
        print(f"  对{i*2+j+1}: A色 {show(a)}   ↔   7色 {show(b)}")

print("\n=== 题面(行41-45, 7 个位置) ===")
q = [cell(g0, 41, 15 + k * 7) for k in range(7)]
for k, m in enumerate(q):
    print(f"  位{k+1}: {show(m)}")

print("\n=== 答案区(行52-56, 7 个位置) ===")
a_ = [cell(g0, 52, 15 + k * 7) for k in range(7)]
for k, m in enumerate(a_):
    print(f"  位{k+1}: {show(m)}")

print("\n=== 实验: 在第1位连按 ACTION1 十次, 看符号怎么变 ===")
ch = clone(game)
seen = []
for i in range(10):
    fr = raw(ch, 1)
    if not fr.frame:
        print(f"  第{i+1}次无帧"); break
    g = np.array(fr.frame[-1])
    m = cell(g, 52, 15)
    seen.append(show(m))
    print(f"  第{i+1:>2}次: {show(m)}   关卡={fr.levels_completed}")
uniq = list(dict.fromkeys(seen))
print(f"  → 出现 {len(uniq)} 种不同符号; 若首尾重复则为循环, 循环长度 = 可选符号数")

print("\n=== 实验: 连按 ACTION4 八次, 看光标怎么走 ===")
ch = clone(game)
prev = g0
for i in range(8):
    fr = raw(ch, 4)
    if not fr.frame:
        break
    g = np.array(fr.frame[-1])
    d = np.argwhere(g != prev)
    cols = sorted({int(c) for r, c in d if 48 <= r <= 49})
    print(f"  第{i+1}次: 行48-49 变动列 {cols}")
    prev = g
