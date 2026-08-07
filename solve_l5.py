"""泛化模型 + 规划 + 真机验证。

泛化(简化偏置): 实测同一格在不同形状下四向落点完全一致 => 移动与形状无关。
形状只在旋转器带(row35 的 col14/19/24)改变, 每次进入顺时针转 90°。
于是把"按(格,形状)记"的碎图压成"按格记"的连通图 + 一条形状规则。
"""
import ast
import heapq
import json
import numpy as np
from wm import Percept, energy, load_env, step

MAXE = 42
LOCK = (10, 54)
ROTATORS = {(35, 14), (35, 19), (35, 24)}
PICKUPS = {(5, 44), (10, 9), (45, 14)}   # B环: 实测回满而非固定加值


def rot_cw(b):
    """3x3 位图顺时针 90°: new[i][j] = old[2-j][i]"""
    out = 0
    for i in range(3):
        for j in range(3):
            if b >> ((2 - j) * 3 + i) & 1:
                out |= 1 << (i * 3 + j)
    return out


def build(path="l5_solution.json"):
    """压成两张表: move[(cell,act)]=cell' (与形状无关), xform[cell][sh]=sh' (落地形变)。"""
    d = json.load(open(path))
    T = {ast.literal_eval(k): ast.literal_eval(v) for k, v in d["transitions"].items()}
    votes, xform, disagree = {}, {}, 0
    for (state, a), v in T.items():
        if not isinstance(v[0], tuple):
            continue
        cell, sh, _ = state
        dst, sh2 = v[0][0], v[0][1]
        votes.setdefault((cell, a), {}).setdefault(dst, 0)
        votes[(cell, a)][dst] += 1
        if dst != cell:                      # 只在真的移动时学落地形变
            xform.setdefault(dst, {})[sh] = sh2
    move = {}
    for key, opts in votes.items():
        if len(opts) > 1:
            disagree += 1
        move[key] = max(opts.items(), key=lambda x: x[1])[0]
    nontrivial = {c: m for c, m in xform.items() if any(k != v for k, v in m.items())}
    print(f"压成 {len(move)} 条格级转移(分歧 {disagree} 条); "
          f"形变格 {len(nontrivial)} 个: {sorted(nontrivial)}")
    return move, xform


def plan(move, xform, start_cell, start_shape, e0, want_shape, forbid=()):
    """状态=(格, 形状, 能量, 是否已取补给); 目标=带 want_shape 站在锁前。"""
    s0 = (start_cell, start_shape, e0, frozenset())
    pq = [(0, s0, [])]
    best = {s0: 0}
    while pq:
        cost, (cell, sh, e, got), path = heapq.heappop(pq)
        if cell == LOCK and (want_shape is None or sh == want_shape):
            return path, e, sh
        for a in (1, 2, 3, 4):
            dst = move.get((cell, a))
            if dst is None or dst in forbid:
                continue
            if dst == cell:
                sh2 = sh
            elif dst in xform and sh in xform[dst]:
                sh2 = xform[dst][sh]          # 学到的落地形变(含旋转器与(10,19)变形器)
            elif dst in ROTATORS and isinstance(sh, int):
                sh2 = rot_cw(sh)
            else:
                sh2 = sh
            e2 = e - 2
            got2 = got
            if dst in PICKUPS and dst not in got:
                e2, got2 = MAXE, got | {dst}
            if e2 < 2:
                continue
            k = (dst, sh2, e2, got2)
            if best.get(k, 1 << 30) <= cost + 1:
                continue
            best[k] = cost + 1
            heapq.heappush(pq, (cost + 1, k, path + [a]))
    return None


def verify(seq):
    """真机逐步执行, 返回 (是否过关, 终态描述)。"""
    game, f = load_env("solutions_l4.json")
    base = f.levels_completed
    p = Percept(np.array(f.frame[-1]))
    for a in seq:
        f = step(game, a)
        if not f.frame:
            return False, "死亡"
        if f.levels_completed > base:
            return True, "过关"
    g = np.array(f.frame[-1])
    return False, f"停在 key={p.key(g)} energy={energy(g)}"


def main():
    move, xform = build()
    all_shapes = set()
    for m in xform.values():
        all_shapes |= set(m) | set(m.values())
    tried = {122, 179, 410, 188}
    todo = sorted(all_shapes - tried)
    print(f"待试形状 {todo}")
    for sh in todo:
        r = plan(move, xform, (40, 49), 410, MAXE, sh)
        if r is None:
            print(f"形状{sh}: 无可行路径")
            continue
        path, e, _ = r
        seq = path + [1]
        ok, msg = verify(seq)
        print(f"形状{sh}: 规划 {len(path)} 步(余能量{e}) -> 真机 {msg}")
        if ok:
            print("FULL L5 SEQ:", ",".join(map(str, seq)))
            json.dump({"level5_seq": seq}, open("l5_seq.json", "w"))
            return


if __name__ == "__main__":
    main()
