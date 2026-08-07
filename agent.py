"""ls20 智能体: 学模型 -> 模型上规划 -> 真机执行并验证 -> 反例修模型。

架构对应文献: 符号感知(帧diff) + 可执行世界模型(Python转移表) + 模型内规划(Dijkstra)
+ 逐步验证(预测与实测不符即反例) + 定向探索(Go-Explore 的"回到边界再探索")。
"""
import argparse
import heapq
import json
from collections import defaultdict
import numpy as np
from wm import Percept, energy, load_env, step

MAXE = 42
ACTIONS = (1, 2, 3, 4)


class Model:
    """(cell, shape, action) -> (cell', shape', dE); 只收非死亡转移。"""

    def __init__(self):
        self.t = {}
        self.counterexamples = []

    def add(self, s, a, s2, dE):
        k = (s, a)
        v = (s2, dE)
        if k in self.t and self.t[k] != v:
            self.counterexamples.append((k, self.t[k], v))
            return False
        self.t[k] = v
        return True

    def get(self, s, a):
        return self.t.get((s, a))

    def known(self, s):
        return [a for a in ACTIONS if (s, a) in self.t]

    def unknown(self, s):
        return [a for a in ACTIONS if (s, a) not in self.t]


def plan(model, start, e0, goal_test, energy_floor=1):
    """Dijkstra over (cell/shape state, energy). 返回 (动作序列, 终态) 或 None。"""
    pq = [(0, start, e0, [])]
    best = {(start, e0): 0}
    while pq:
        cost, s, e, path = heapq.heappop(pq)
        if goal_test(s, e, path):
            return path, s, e
        for a in ACTIONS:
            r = model.get(s, a)
            if r is None:
                continue
            s2, dE = r
            e2 = min(MAXE, e + dE)
            if e2 < energy_floor:
                continue
            k = (s2, e2)
            if best.get(k, 1 << 30) <= cost + 1:
                continue
            best[k] = cost + 1
            heapq.heappush(pq, (cost + 1, s2, e2, path + [a]))
    return None


class Runner:
    """真模拟器句柄: 执行动作并给出抽象观测; 死亡自动重开。"""

    def __init__(self, prefix_file):
        self.prefix_file = prefix_file
        self.restart()

    def restart(self):
        self.game, f = load_env(self.prefix_file)
        g0 = np.array(f.frame[-1])
        self.p = Percept(g0)
        f = step(self.game, 1)          # 一步用于标定感知
        self.p.calibrate(g0, np.array(f.frame[-1]))
        self.base_level = f.levels_completed
        self.f = f
        return self.obs()

    def obs(self):
        if not self.f.frame:           # GAME_OVER 后引擎返回空帧
            return None, 0
        g = np.array(self.f.frame[-1])
        return self.p.state(g, self.f.levels_completed), energy(g)

    def act(self, a):
        self.f = step(self.game, a)
        st, lv = self.f.state.name, self.f.levels_completed
        if not self.f.frame or st == "GAME_OVER":
            s, e = self.restart()
            return s, e, "GAME_OVER", lv
        s, e = self.obs()
        return s, e, st, lv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix-file", default="solutions_l4.json")
    ap.add_argument("--budget", type=int, default=6000, help="真机动作预算")
    ap.add_argument("--out", default="l5_solution.json")
    args = ap.parse_args()

    model = Model()
    r = Runner(args.prefix_file)
    s, e = r.obs()
    start_state = s
    used = 0
    deaths = 0
    win_path = None
    tried_from_start = defaultdict(int)

    start_cell = s[0] if s else None
    while used < args.budget:
        # 1) 目标: 若模型里已知通往过关的动作, 直接规划过去
        # 2) 否则: 规划到"存在未试动作"的最近状态, 去试它
        target = plan(model, s, e,
                      lambda st, en, p: len(model.unknown(st)) > 0 and len(p) > 0)
        if target is None:
            unk = model.unknown(s)
            if unk:
                path, act = [], unk[0]
            else:
                # 无处可探: 消耗能量自杀重生, 换个起点分支
                path, act = [], ACTIONS[tried_from_start[s] % 4]
                tried_from_start[s] += 1
        else:
            path, tstate, tenergy = target
            act = model.unknown(tstate)[0]

        ok = True
        for a in path:                      # 执行计划并逐步验证
            pred = model.get(s, a)
            s2, e2, st, lv = r.act(a)
            used += 1
            if lv > r.base_level:
                win_path = "LEVELUP during plan"
                break
            if st == "GAME_OVER" or e2 > e:  # 死亡重生
                deaths += 1
                s, e = r.obs()
                ok = False
                break
            if pred and pred[0] != s2:
                model.counterexamples.append(((s, a), pred, (s2, e2 - e)))
                model.t[(s, a)] = (s2, e2 - e)
                s, e = s2, e2
                ok = False
                break
            s, e = s2, e2
        if win_path:
            break
        if not ok:
            continue

        s_before, e_before = s, e
        s2, e2, st, lv = r.act(act)
        used += 1
        if lv > r.base_level:
            print(f"*** LEVEL UP at action {used}! from {s_before} act {act}", flush=True)
            model.add(s_before, act, "GOAL", e2 - e_before)
            win_path = True
            break
        # 死亡 = 被传送回关卡起点; 能量上升但位置正常 = 捡到补给, 必须入模型
        if st == "GAME_OVER" or (e2 > e_before and s2 and s2[0] == start_cell):
            deaths += 1
            s, e = r.obs()
            continue
        model.add(s_before, act, s2, e2 - e_before)
        s, e = s2, e2

        if used % 500 == 0:
            cells = {k[0][0] for k in model.t}
            print(f"used {used}: {len(model.t)} transitions, {len(cells)} cells, "
                  f"{deaths} deaths, {len(model.counterexamples)} counterexamples", flush=True)

    cells = sorted({k[0][0] for k in model.t})
    print(f"DONE used={used} deaths={deaths} transitions={len(model.t)} "
          f"cells={len(cells)} counterexamples={len(model.counterexamples)} win={bool(win_path)}")
    print("cells:", cells)
    json.dump({"transitions": {repr(k): repr(v) for k, v in model.t.items()},
               "counterexamples": [repr(c) for c in model.counterexamples[:30]],
               "cells": [repr(c) for c in cells], "win": bool(win_path)},
              open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
