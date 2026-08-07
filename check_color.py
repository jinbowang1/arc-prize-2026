"""验证: 过关是否要求形状+颜色都匹配。"""
import json, numpy as np, arc_agi
from arcengine import GameAction
from validate_lock import find_locks, show
A = {1: GameAction.ACTION1, 2: GameAction.ACTION2, 3: GameAction.ACTION3, 4: GameAction.ACTION4}

def panel(g):
    """左下面板: 返回 (形状bits, 颜色)。"""
    bits, cols = 0, set()
    for i in range(3):
        for j in range(3):
            blk = g[55+i*2:57+i*2, 3+j*2:5+j*2]
            v = set(blk.flatten().tolist()) - {5}
            if v:
                bits |= 1 << (i*3+j)
                cols |= v
    return bits, (sorted(cols)[0] if cols else None)

seq = json.load(open("solutions.json"))["seq"]
arc = arc_agi.Arcade(); env = arc.make("ls20"); f = env.reset()
lv = 0
lock = find_locks(np.array(f.frame[-1]))
for a in seq:
    pb, pc = panel(np.array(f.frame[-1]))
    f = env.step(A[a])
    if f.levels_completed > lv:
        lv = f.levels_completed
        for (_, _, lc, lb) in lock:
            print(f"L{lv}: 面板 形状{pb} 色{pc}  |  锁 形状{lb} 色{lc}  "
                  f"{'形状✓' if pb==lb else '形状✗'} {'颜色✓' if pc==lc else '颜色✗'}")
        lock = find_locks(np.array(f.frame[-1]))
g = np.array(f.frame[-1])
pb, pc = panel(g)
print(f"L{lv+1} 当前: 面板 形状{pb} 色{pc}")
for (_, _, lc, lb) in find_locks(g):
    print(f"L{lv+1} 锁要求: 形状{lb} 色{lc}")
