"""ls20 试玩器: 重放前缀动作序列后, BFS 规划钥匙块走到目标, 执行并报告。"""
import argparse
import numpy as np
import arc_agi
from arcengine import GameAction

A = {1: GameAction.ACTION1, 2: GameAction.ACTION2, 3: GameAction.ACTION3, 4: GameAction.ACTION4}
DELTA = {1: (-5, 0), 2: (5, 0), 3: (0, -5), 4: (0, 5)}
CHARS = ".123456789ABCDEF"


def key_pos(g):
    pos = np.argwhere(g == 12)
    pos = pos[pos[:, 0] < 50]
    return (int(pos[:, 0].min()), int(pos[:, 1].min())) if len(pos) else None


def passable(g, r, c, blocked_colors, avoid):
    if r < 0 or c < 0 or r + 5 > 64 or c + 5 > 64 or (r, c) in avoid:
        return False
    win = g[r:r + 5, c:c + 5]
    return not any((win == b).any() for b in blocked_colors)


def bfs(g, start, goal, blocked_colors, avoid):
    from collections import deque
    q = deque([(start, [])])
    seen = {start}
    while q:
        (r, c), path = q.popleft()
        if (r, c) == goal:
            return path
        for a, (dr, dc) in DELTA.items():
            nr, nc = r + dr, c + dc
            if (nr, nc) not in seen and passable(g, nr, nc, blocked_colors, avoid):
                seen.add((nr, nc))
                q.append(((nr, nc), path + [a]))
    return None


def render(g):
    return "\n".join("".join(CHARS[v] for v in row) for row in g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="", help="先重放的动作序列")
    ap.add_argument("--goal", required=True, help="r,c 目标格左上角")
    ap.add_argument("--blocked", default="4", help="视为墙的颜色")
    ap.add_argument("--avoid", default="", help="额外避开的格子 r,c;r,c")
    ap.add_argument("--dry", action="store_true", help="只规划不执行")
    args = ap.parse_args()

    goal = tuple(int(x) for x in args.goal.split(","))
    blocked = [int(x) for x in args.blocked.split(",")]
    avoid = set()
    if args.avoid:
        for part in args.avoid.split(";"):
            r, c = part.split(",")
            avoid.add((int(r), int(c)))

    arc = arc_agi.Arcade()
    env = arc.make("ls20")
    f = env.reset()
    for a in [x for x in args.prefix.split(",") if x.strip()]:
        f = env.step(A[int(a)])
    g = np.array(f.frame[-1])
    start = key_pos(g)
    print(f"after prefix({len([x for x in args.prefix.split(',') if x.strip()])}): "
          f"lv={f.levels_completed} key at {start}")
    path = bfs(g, start, goal, blocked, avoid)
    if path is None:
        print("NO PATH")
        return
    full = args.prefix + ("," if args.prefix else "") + ",".join(map(str, path))
    print(f"path ({len(path)} moves): {','.join(map(str, path))}")
    print(f"full seq: {full}")
    if args.dry:
        return
    for a in path:
        f = env.step(A[a])
    g = np.array(f.frame[-1])
    print(f"FINAL lv={f.levels_completed}/{f.win_levels} state={f.state.name} key={key_pos(g)}")
    print(render(g))


if __name__ == "__main__":
    main()
