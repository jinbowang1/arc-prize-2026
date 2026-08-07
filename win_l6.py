"""L6: 先攻上锁(格35,54, 需 形状413+色8), 从 (30,54) 向下进入。

L6 机关分区: row40 纯旋转带 / row10 换族带 / rows20-30 × cols19-29 调色区(14→8→12→9)
起始 形状371 色14 => 旋转两次得413, 调色一次得8。
"""
import json
import numpy as np
import solve_l5
from solve_l5 import build, plan
from wm import Percept, energy, load_env, panel_color, shape_bits, step

ROTS = {(40, c) for c in (14, 19, 24, 29, 34)}
XFORMS = {(10, c) for c in (14, 19, 24, 29, 34)}
PALETTE = {(r, c) for r in (20, 25, 30) for c in (19, 24, 29)}
move, xform = build("l6_model.json")


class Run:
    def __init__(self):
        self.game, self.f = load_env("solutions_l5.json")
        self.base = self.f.levels_completed
        self.p = Percept(np.array(self.f.frame[-1]))
        self.seq = []

    def st(self):
        g = np.array(self.f.frame[-1])
        return self.p.key(g), shape_bits(g), panel_color(g), energy(g)

    def do(self, a):
        self.f = step(self.game, a)
        self.seq.append(a)
        return bool(self.f.frame)

    def nav(self, to, forbid=()):
        cell, sh, col, e = self.st()
        solve_l5.LOCK = to
        r = plan(move, xform, cell, (sh, col), e, None, forbid=forbid)
        if r is None:
            return False
        for a in r[0]:
            if not self.do(a):
                return False
        return True

    def wiggle(self, pair, want, idx, limit=12):
        """在机关格与邻格间来回, 直到目标维达成。idx: 1=形状 2=颜色"""
        for i in range(limit):
            if self.st()[idx] == want:
                return True
            if not self.do(pair[i % 2]):
                return False
        return self.st()[idx] == want


def main():
    r = Run()
    print("起始:", r.st())

    # 一次性规划: 到达 (30,54) 时形状=413 且颜色=8, 沿途机关由模型自行调度
    cell, sh, col, e = r.st()
    solve_l5.LOCK = (30, 54)
    res = plan(move, xform, cell, (sh, col), e, (413, 8))
    if res is None:
        return print("模型内无满足双条件的路径")
    print(f"规划 {len(res[0])} 步, 余能量 {res[1]}")
    for a in res[0]:
        if not r.do(a):
            return print("中途死亡")
    print("锁前:", r.st())
    r.do(2)                                   # 向下进入上锁
    if r.f.frame and r.f.levels_completed > r.base:
        print(f"*** L6 通关! 本关 {len(r.seq)} 步")
        json.dump({"level6_seq": r.seq}, open("l6_seq.json", "w"))
    else:
        print("上锁未开, 现在", r.st() if r.f.frame else "死亡")


if __name__ == "__main__":
    main()
