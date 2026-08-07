"""L5 完整解: 形状 413 + 颜色 8 同时满足, 再进锁。

L5 的新机制 = 颜色维度(L1-L4 恒为9故从未暴露)。过关要求形状与颜色都匹配。
  形状: 变形器(10,19)按环换族 -> 旋转带(row35)转向
  颜色: 调色板(25,29)按 12->9->14->8 循环
拿到某维后, 导航必须禁行会破坏它的格子。
"""
import json
import numpy as np
import solve_l5
from solve_l5 import build, plan
from wm import Percept, energy, load_env, shape_bits, step
from check_color import panel

XFORMER, PALETTE = (10, 19), (25, 29)
ROT = solve_l5.ROTATORS
TARGET_SHAPE, TARGET_COLOR = 413, 8
move, xform = build()


class Run:
    def __init__(self):
        self.game, self.f = load_env("solutions_l4.json")
        self.base = self.f.levels_completed
        self.p = Percept(np.array(self.f.frame[-1]))
        self.seq = []

    @property
    def g(self):
        return np.array(self.f.frame[-1])

    def state(self):
        g = self.g
        sh, col = panel(g)
        return self.p.key(g), sh, col, energy(g)

    def do(self, a):
        self.f = step(self.game, a)
        self.seq.append(a)
        return bool(self.f.frame) and self.f.levels_completed == self.base

    def nav(self, to, forbid=()):
        cell, sh, _, e = self.state()
        solve_l5.LOCK = to
        r = plan(move, xform, cell, sh, e, None, forbid=forbid)
        if r is None:
            return False
        for a in r[0]:
            if not self.do(a):
                return False
        return True


def main():
    r = Run()
    # ---- 阶段1: 形状 -> 413 ----
    if not r.nav((10, 14)):
        return print("到不了变形器")
    for _ in range(8):
        r.do(4)
        if r.state()[1] == 371:
            break
        r.do(3)
    print("阶段1a 变形器:", r.state())
    if r.state()[1] != 371:
        return print("没拿到371")

    if not r.nav((10, 9), forbid=(XFORMER,)):
        return print("到不了补给")
    if not r.nav((35, 19), forbid=(XFORMER,)):
        return print("到不了旋转带")
    for i in range(16):
        r.do(3 if i % 2 == 0 else 4)
        if r.state()[1] == TARGET_SHAPE:
            break
    print("阶段1b 旋转带:", r.state())
    if r.state()[1] != TARGET_SHAPE:
        return print("没凑出413")

    # ---- 阶段2: 颜色 -> 8 (禁行一切改形状的格) ----
    F_shape = (XFORMER,) + tuple(ROT)
    if not r.nav(PALETTE, forbid=F_shape):
        return print("到不了调色板")
    print("阶段2a 到调色板:", r.state())
    for i in range(8):
        if r.state()[2] == TARGET_COLOR:
            break
        r.do(4)
        r.do(3)
    print("阶段2b 调色完成:", r.state())
    if r.state()[2] != TARGET_COLOR:
        return print("没拿到色8")

    # ---- 阶段3: 保住两维去锁前 ----
    F_all = F_shape + (PALETTE,)
    for tgt in [(5, 44), (10, 54)]:
        if not r.nav(tgt, forbid=F_all):
            return print(f"到不了 {tgt}, 当前 {r.state()}")
        print(f"阶段3 到 {tgt}:", r.state())

    cell, sh, col, e = r.state()
    print(f"锁前: 位置{cell} 形状{sh}(需{TARGET_SHAPE}) 颜色{col}(需{TARGET_COLOR}) 能量{e}")
    r.do(1)
    if r.f.frame and r.f.levels_completed > r.base:
        print(f"*** L5 通关!!! 本关 {len(r.seq)} 步")
        json.dump({"level5_seq": r.seq}, open("l5_seq.json", "w"))
    else:
        print("仍未开锁")


if __name__ == "__main__":
    main()
