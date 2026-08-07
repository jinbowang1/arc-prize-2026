"""在真模拟器里行走, 学出格级转移表; 矛盾即抽象漏了隐藏变量, 记录为反例。

用法: OPERATION_MODE=OFFLINE uv run python learn.py --steps 3000
"""
import argparse
import json
from collections import defaultdict
import numpy as np
from wm import Percept, energy, load_env, step


def observe(p, f):
    g = np.array(f.frame[-1])
    return p.state(g, f.levels_completed), energy(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--prefix-file", default="solutions.json")
    ap.add_argument("--out", default="model_l5.json")
    args = ap.parse_args()

    game, f = load_env(args.prefix_file)
    g0 = np.array(f.frame[-1])
    p = Percept(g0)
    f = step(game, 1)
    p.calibrate(g0, np.array(f.frame[-1]))
    base_level = f.levels_completed

    trans = {}            # (state, action) -> (next_state, dE, flag)
    tries = defaultdict(int)
    conflicts = []
    deaths = 0
    levelups = 0

    s, e = observe(p, f)
    for i in range(args.steps):
        # 探索策略: 选当前状态下尝试次数最少的动作
        a = min((1, 2, 3, 4), key=lambda x: tries[(s, x)])
        tries[(s, a)] += 1
        f = step(game, a)
        s2, e2 = observe(p, f)
        dE = e2 - e
        st = f.state.name

        if f.levels_completed > base_level:
            levelups += 1
            trans[repr((s, a))] = [repr(s2), dE, "LEVELUP"]
            break
        if st == "GAME_OVER":
            deaths += 1
            game, f = load_env(args.prefix_file)
            g0 = np.array(f.frame[-1])
            p = Percept(g0)
            f = step(game, 1)
            p.calibrate(g0, np.array(f.frame[-1]))
            s, e = observe(p, f)
            continue

        key = repr((s, a))
        rec = [repr(s2), dE, st]
        if key in trans and trans[key][0] != rec[0]:
            conflicts.append({"from": key, "seen": trans[key], "now": rec,
                              "energy_before": e})
        else:
            trans[key] = rec
        if e2 > e:
            print(f"  step{i}: ENERGY PICKUP at {s} -> {s2}, +{dE}", flush=True)
        if e2 < e - 5:      # 死亡回起点
            deaths += 1
        s, e = s2, e2
        if (i + 1) % 500 == 0:
            print(f"step {i+1}: {len(trans)} transitions, {len(set(k.split(',')[0] for k in trans))} states, "
                  f"{len(conflicts)} conflicts, {deaths} deaths", flush=True)

    json.dump({"transitions": trans, "conflicts": conflicts[:40],
               "n_conflicts": len(conflicts), "deaths": deaths,
               "levelups": levelups}, open(args.out, "w"), indent=1)
    states = set()
    for k in trans:
        states.add(k.rsplit(", ", 1)[0])
    print(f"DONE: {len(trans)} transitions over ~{len(states)} states, "
          f"{len(conflicts)} conflicts, {deaths} deaths, {levelups} levelups")
    for c in conflicts[:5]:
        print("  CONFLICT", c["from"], "seen", c["seen"], "now", c["now"])


if __name__ == "__main__":
    main()
