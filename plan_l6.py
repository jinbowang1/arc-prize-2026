"""L6 专用规划器: 机关规律写成规则而非查表(查表太稀疏, 组合一多就断链)。

规则: row40 = 旋转带(rot_cw) / rows20-30×cols19-29 = 调色环(14→8→12→9→14)
      row10 = 换族带(只能查表, 用模型学到的映射)
"""
import ast
import heapq
import json
from solve_l5 import rot_cw

MAXE = 42
ROTS = {(40, c) for c in (14, 19, 24, 29, 34)}
XFORMS = {(10, c) for c in (14, 19, 24, 29, 34)}
PALETTE = {(r, c) for r in (20, 25, 30) for c in (19, 24, 29)}
CYCLE = {14: 8, 8: 12, 12: 9, 9: 14}


def load(path="l6_model.json"):
    d = json.load(open(path))
    T = {ast.literal_eval(k): ast.literal_eval(v) for k, v in d["transitions"].items()}
    votes, fam, pickups = {}, {}, {}
    for (state, a), v in T.items():
        if not isinstance(v[0], tuple):
            continue
        cell = state[0]
        sh, col = state[1]
        dst, (sh2, col2) = v[0][0], v[0][1]
        votes.setdefault((cell, a), {}).setdefault(dst, 0)
        votes[(cell, a)][dst] += 1
        if dst in XFORMS and dst != cell:
            fam[(dst, sh)] = sh2
        if v[1] > 0 and dst != cell:
            pickups[dst] = max(pickups.get(dst, 0), v[1])
    move = {k: max(o.items(), key=lambda x: x[1])[0] for k, o in votes.items()}
    return move, fam, set(pickups)


def step_rule(cell, sh, col, dst, fam):
    """落地后的 (形状, 颜色)。返回 None 表示换族表里没这条。"""
    if dst == cell:
        return sh, col
    if dst in ROTS:
        return rot_cw(sh), col
    if dst in PALETTE:
        return sh, CYCLE.get(col, col)
    if dst in XFORMS:
        r = fam.get((dst, sh))
        return (r, col) if r is not None else None
    return sh, col


def plan(move, fam, pickups, start, e0, goal_cell, goal_sh=None, goal_col=None, forbid=(), min_e=0):
    s0 = (start[0], start[1], start[2], e0, frozenset())
    pq = [(0, s0, [])]
    best = {s0: 0}
    while pq:
        cost, (cell, sh, col, e, got), path = heapq.heappop(pq)
        if cell == goal_cell and (goal_sh is None or sh == goal_sh) and (goal_col is None or col == goal_col) and e >= min_e:
            return path, e
        for a in (1, 2, 3, 4):
            dst = move.get((cell, a))
            if dst is None or dst in forbid:
                continue
            nxt = step_rule(cell, sh, col, dst, fam)
            if nxt is None:
                continue
            sh2, col2 = nxt
            e2, got2 = e - 2, got
            if dst in pickups and dst not in got:
                e2, got2 = MAXE, got | {dst}
            if e2 < 2:
                continue
            k = (dst, sh2, col2, e2, got2)
            if best.get(k, 1 << 30) <= cost + 1:
                continue
            best[k] = cost + 1
            heapq.heappush(pq, (cost + 1, k, path + [a]))
    return None
