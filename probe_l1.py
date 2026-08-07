"""L1 对照实验: 带错误形状走到锁前, 看门的反应是否同为零消耗。"""
import numpy as np, arc_agi
from arcengine import GameAction
from wm import shape_bits, energy, Percept
from validate_lock import find_locks, show
A = {1: GameAction.ACTION1, 2: GameAction.ACTION2, 3: GameAction.ACTION3, 4: GameAction.ACTION4}
arc = arc_agi.Arcade(); env = arc.make("ls20"); f = env.reset()
g = np.array(f.frame[-1]); p = Percept(g)
print("L1 锁显示:", [(b) for *_ , b in find_locks(g)], "| 起始形状", shape_bits(g))
# 直上, 绕开旋转器
for i in range(9):
    e0 = energy(np.array(f.frame[-1]))
    f = env.step(A[1])
    g = np.array(f.frame[-1])
    print(f"  上{i+1}: 位置={p.key(g)} 形状={shape_bits(g)} 能量变化={energy(g)-e0} lv={f.levels_completed}")
    if f.levels_completed > 0:
        print("  (直上就通关了)"); break
