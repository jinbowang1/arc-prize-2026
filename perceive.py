"""感知层: 靠帧 diff 找真正在动的对象, 不靠图案猜。

用法: OPERATION_MODE=OFFLINE uv run python perceive.py '[1,3,3,1]'
"""
import json
import sys
import numpy as np
import arc_agi
from arcengine import GameAction

A = {1: GameAction.ACTION1, 2: GameAction.ACTION2, 3: GameAction.ACTION3, 4: GameAction.ACTION4}
CH = ".123456789ABCDEF"


def blocks(g, colors=None):
    """把 64x64 网格按 5x5 对齐块扫描, 返回 {(r,c): 主色} 仅限纯色块。"""
    out = {}
    for r in range(0, 60):
        for c in range(0, 60):
            w = g[r:r + 5, c:c + 5]
            v = set(w.flatten().tolist())
            if len(v) == 1:
                out[(r, c)] = v.pop()
    return out


def moving_object(prev, cur):
    """返回 (消失区域, 出现区域): 用色块 diff 定位移动主体。"""
    d = np.argwhere(prev != cur)
    if len(d) == 0:
        return None
    regions = {}
    for r, c in d:
        key = (int(prev[r, c]), int(cur[r, c]))
        regions.setdefault(key, []).append((int(r), int(c)))
    out = []
    for (pv, cv), cells in sorted(regions.items(), key=lambda x: -len(x[1])):
        rs = [x[0] for x in cells]
        cs = [x[1] for x in cells]
        out.append(f"{pv}->{cv} x{len(cells)} rows{min(rs)}-{max(rs)} cols{min(cs)}-{max(cs)}")
    return "; ".join(out[:6])


def main():
    seq = json.loads(sys.argv[1]) if len(sys.argv) > 1 else []
    prefix = json.load(open("solutions.json"))["seq"]
    arc = arc_agi.Arcade()
    env = arc.make("ls20")
    env.reset()
    for a in prefix:
        f = env.step(A[a])
    prev = np.array(f.frame[-1])
    print(f"L{f.levels_completed + 1} start, nframes={len(f.frame)}")
    for i, a in enumerate(seq):
        f = env.step(A[a])
        cur = np.array(f.frame[-1])
        print(f"s{i+1} a{a} lv={f.levels_completed} frames={len(f.frame)} | {moving_object(prev, cur)}")
        prev = cur


if __name__ == "__main__":
    main()
