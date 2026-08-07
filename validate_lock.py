"""交叉验证锁显示解码规则: 通关瞬间的形状是否等于该关锁上显示的图案。"""
import json
import numpy as np
import arc_agi
from arcengine import GameAction
from wm import shape_bits

A = {1: GameAction.ACTION1, 2: GameAction.ACTION2, 3: GameAction.ACTION3, 4: GameAction.ACTION4}


def show(b):
    return "/".join("".join("X" if b >> (i * 3 + j) & 1 else "." for j in range(3))
                    for i in range(3))


def find_locks(g):
    """7x7 全色5边框的显示屏, 解正中 3x3(每格1像素)。"""
    out = []
    for r in range(0, 57):
        for c in range(0, 57):
            w = g[r:r + 7, c:c + 7]
            border = np.concatenate([w[0], w[6], w[:, 0], w[:, 6]])
            if not (border == 5).all():
                continue
            core = w[2:5, 2:5]
            vals = set(core.flatten().tolist()) - {5}
            if len(vals) != 1:
                continue
            col = vals.pop()
            bits = sum(1 << (i * 3 + j) for i in range(3) for j in range(3)
                       if core[i, j] == col)
            out.append((r, c, col, bits))
    return out


if __name__ == "__main__":
    seq = json.load(open("solutions.json"))["seq"]
    arc = arc_agi.Arcade()
    env = arc.make("ls20")
    f = env.reset()
    lv = 0
    target = find_locks(np.array(f.frame[-1]))   # L1 开局读锁
    for a in seq:
        prev_shape = shape_bits(np.array(f.frame[-1]))
        f = env.step(A[a])
        if f.levels_completed > lv:
            lv = f.levels_completed
            tb = [b for (_, _, _, b) in target]
            ok = "✓ 一致" if prev_shape in tb else "✗ 不一致"
            print(f"L{lv}: 开局锁显示 {tb} vs 通关时形状 {prev_shape} = {show(prev_shape)}  {ok}")
            target = find_locks(np.array(f.frame[-1]))   # 下一关开局读锁

    g = np.array(f.frame[-1])
    print(f"\n当前 L{lv+1} 的锁显示(即待求目标):")
    for (r, c, col, bits) in find_locks(g):
        print(f"    @({r},{c}) 色{col}: {bits} = {show(bits)}")
    print(f"当前形状 {shape_bits(g)} = {show(shape_bits(g))}")
