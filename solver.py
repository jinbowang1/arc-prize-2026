"""ls20 状态图 BFS 求解器 v2: 单游戏实例 + pickle 增量快照, 逐关求最短动作序列。

快照只含动态字段 + 当前关卡条目(~22KB), _clean_levels 与非当前关共享引用。
逐关结果追加写入 solutions.json。
"""
import argparse
import copy
import json
import time
import numpy as np
import arc_agi
from arcengine import GameAction, ActionInput

ACTS = {a: getattr(GameAction, f"ACTION{a}") for a in (1, 2, 3, 4)}


def step(game, a):
    return game.perform_action(ActionInput(id=ACTS[a], data={}), raw=True)


def clone(game, clean):
    """唯一安全模式: 整对象 deepcopy 后用克隆(单一 memo 保全部交叉引用),
    绝不原地恢复。只读的 _clean_levels 借出还回, 全部克隆共享同一份。"""
    game._clean_levels = None
    g2 = copy.deepcopy(game)
    game._clean_levels = clean
    g2._clean_levels = clean
    return g2


def solve_level(game, cur_level, max_nodes=2000000, log_every=20):
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
                h = np.asarray(f.frame[-1], dtype=np.uint8).tobytes()
                if h in seen:
                    continue
                seen.add(h)
                nxt.append((child, path + [a]))
        if nodes > max_nodes:
            print("  node cap hit")
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
    ap.add_argument("--prefix", default="")
    ap.add_argument("--levels", type=int, default=7)
    ap.add_argument("--out", default="solutions.json")
    args = ap.parse_args()

    arc = arc_agi.Arcade()
    env = arc.make(args.game)
    f = env.reset()
    game = env._game
    prefix = [int(x) for x in args.prefix.split(",") if x.strip()]
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
        # 在真实 game 上重放本关解, 推进到下一关
        for a in seq:
            f = step(game, a)
        assert f.levels_completed > lv, "replay diverged!"
        lv = f.levels_completed
        all_seq += seq
        print(f"LEVEL {lv}: {len(seq)} actions in {time.time()-t0:.0f}s: "
              f"{','.join(map(str, seq))}", flush=True)
        with open(args.out, "w") as fh:
            json.dump({"game": args.game, "seq": all_seq, "levels_done": lv}, fh)
    print("FULL SEQ:", ",".join(map(str, all_seq)), flush=True)
    # 终验: 全新环境从头重放
    env2 = arc.make(args.game)
    f2 = env2.reset()
    for a in all_seq:
        f2 = env2.step({1: GameAction.ACTION1, 2: GameAction.ACTION2,
                        3: GameAction.ACTION3, 4: GameAction.ACTION4}[a])
    print(f"VERIFY on fresh env: levels={f2.levels_completed} state={f2.state.name}", flush=True)


if __name__ == "__main__":
    main()
