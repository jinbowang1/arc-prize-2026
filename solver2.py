"""ls20 求解器 v5: 语义状态哈希 + 能量分桶。

v4 的致命问题: 哈希用整帧字节, 而能量条每步递减、顶部计时器按节律走,
同一位置在不同步数下永不相等 → 去重失效, BFS 退化成近似穷举。
v5 只哈希地图与形状面板(掩掉 rows 61-62 的能量条/步数计数器),
能量单独按桶粗粒度参与, 兼顾正确性与去重率。
"""
import argparse
import copy
import json
import time
import numpy as np
import arc_agi
from arcengine import GameAction, ActionInput

ACTS = {a: getattr(GameAction, f"ACTION{a}") for a in (1, 2, 3, 4)}
HUD_ROWS = slice(61, 63)   # 能量条 + 步数计数器: 纯计步噪声, 不进哈希


def step(game, a):
    return game.perform_action(ActionInput(id=ACTS[a], data={}), raw=True)


def clone(game, clean):
    """整对象 deepcopy(单一 memo 保全交叉引用), 只读的 _clean_levels 借出还回。"""
    game._clean_levels = None
    g2 = copy.deepcopy(game)
    game._clean_levels = clean
    g2._clean_levels = clean
    return g2


def sem_hash(frame, bucket=6):
    """语义哈希: 掩掉 HUD 计步噪声, 能量按桶保留(防跨能量档误合并)。"""
    g = np.array(frame, dtype=np.uint8)
    energy = int((g[61] == 11).sum())
    m = g.copy()
    m[HUD_ROWS] = 0
    return m.tobytes() + bytes([energy // bucket])


def solve_level(game, cur_level, max_nodes=3000000, log_every=10):
    clean = game._clean_levels
    frontier = [(game, [])]
    seen = set()
    nodes = 0
    depth = 0
    t0 = time.time()
    while frontier:
        nxt = []
        for base, path in frontier:
            for a in (1, 2, 3, 4):
                child = clone(base, clean)
                f = step(child, a)
                nodes += 1
                if f.levels_completed > cur_level or f.state.name == "WIN":
                    return path + [a]
                if f.state.name == "GAME_OVER":
                    continue
                h = sem_hash(f.frame[-1])
                if h in seen:
                    continue
                seen.add(h)
                nxt.append((child, path + [a]))
        if nodes > max_nodes:
            print("  node cap hit", flush=True)
            return None
        frontier = nxt
        depth += 1
        if depth % log_every == 0:
            print(f"  depth {depth}, frontier {len(frontier)}, nodes {nodes}, "
                  f"{nodes/(time.time()-t0):.0f} n/s", flush=True)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="ls20")
    ap.add_argument("--prefix-file", default="solutions.json")
    ap.add_argument("--levels", type=int, default=7)
    ap.add_argument("--out", default="solutions.json")
    args = ap.parse_args()

    prefix = json.load(open(args.prefix_file))["seq"]
    arc = arc_agi.Arcade()
    env = arc.make(args.game)
    f = env.reset()
    game = env._game
    for a in prefix:
        f = step(game, a)
    lv = f.levels_completed
    print(f"prefix {len(prefix)} actions -> level {lv}", flush=True)

    all_seq = list(prefix)
    while lv < args.levels:
        t0 = time.time()
        seq = solve_level(game, lv)
        if seq is None:
            print(f"LEVEL {lv+1}: UNSOLVED", flush=True)
            break
        for a in seq:
            f = step(game, a)
        assert f.levels_completed > lv, "replay diverged!"
        lv = f.levels_completed
        all_seq += seq
        print(f"LEVEL {lv}: {len(seq)} actions in {time.time()-t0:.0f}s: "
              f"{','.join(map(str, seq))}", flush=True)
        json.dump({"game": args.game, "seq": all_seq, "levels_done": lv},
                  open(args.out, "w"))
    print("FULL SEQ:", ",".join(map(str, all_seq)), flush=True)


if __name__ == "__main__":
    main()
