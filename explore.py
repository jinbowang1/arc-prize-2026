"""ls20 探索驱动: 从 reset 整序列重放, 打印 ASCII 网格与逐步 diff 摘要。

用法:
  uv run python explore.py --game ls20 --actions "1,1,2,3"   # 重放并显示最终帧
  uv run python explore.py --game ls20 --actions "" --show-all  # 每步都打印帧
颜色 0-15 映射到字符, '.' = 0。
"""
import argparse
import numpy as np
import arc_agi
from arcengine import GameAction

CHARS = ".123456789ABCDEF"

ACT = {"1": GameAction.ACTION1, "2": GameAction.ACTION2,
       "3": GameAction.ACTION3, "4": GameAction.ACTION4,
       "5": GameAction.ACTION5, "R": GameAction.RESET}


def render(grid, crop=None):
    g = np.array(grid)
    if crop:
        r0, r1, c0, c1 = crop
        g = g[r0:r1, c0:c1]
    lines = []
    for i, row in enumerate(g):
        lines.append("".join(CHARS[v] for v in row))
    return "\n".join(lines)


def diff_summary(prev, cur):
    p, c = np.array(prev), np.array(cur)
    changed = np.argwhere(p != c)
    if len(changed) == 0:
        return "no change"
    rs, cs = changed[:, 0], changed[:, 1]
    pairs = {}
    for r, cc in changed:
        key = (int(p[r, cc]), int(c[r, cc]))
        pairs[key] = pairs.get(key, 0) + 1
    return (f"{len(changed)} cells, rows {rs.min()}-{rs.max()} cols {cs.min()}-{cs.max()}, "
            f"transitions {dict(sorted(pairs.items(), key=lambda x: -x[1]))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="ls20")
    ap.add_argument("--actions", default="")
    ap.add_argument("--show-all", action="store_true")
    ap.add_argument("--crop", default=None, help="r0,r1,c0,c1")
    ap.add_argument("--save", default=None, help="保存最终帧到 .npy")
    ap.add_argument("--quiet-steps", action="store_true", help="不打印逐步 diff")
    args = ap.parse_args()

    crop = tuple(int(x) for x in args.crop.split(",")) if args.crop else None
    arc = arc_agi.Arcade()
    env = arc.make(args.game)
    f = env.reset()
    prev = np.array(f.frame[-1])
    print(f"RESET  state={f.state.name} levels={f.levels_completed}/{f.win_levels}")

    seq = [a for a in args.actions.split(",") if a.strip()]
    for i, a in enumerate(seq):
        f = env.step(ACT[a.strip()])
        cur = np.array(f.frame[-1])
        ev = ""
        if f.state.name not in ("NOT_FINISHED", "NOT_PLAYED"):
            ev = f" <<< {f.state.name}"
        if not args.quiet_steps:
            print(f"step{i+1:3d} act{a} lv={f.levels_completed} {diff_summary(prev, cur)}{ev}")
        prev = cur

    print(f"FINAL  state={f.state.name} levels={f.levels_completed}/{f.win_levels} nframes={len(f.frame)}")
    print(render(prev, crop))
    if args.save:
        np.save(args.save, prev)


if __name__ == "__main__":
    main()
