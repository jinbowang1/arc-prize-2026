"""对照实验: 继承的 object_to_object 在 ls20 L2 真解轨迹上, h 是怎么走的?

回答一个问题: 30s 最佳优先 h 卡在 10 不动, 是**搜索预算不够**(h 沿真解单调降,
只是搜不到那么深), 还是**启发式在解的前段就是平的**(两段式结构, 缺"角色→钥匙"
那一段的梯度)?

做法: 完全复刻 agent 的学习路径(L1 解出→fit→learn_goals→取第一条关系型),
然后沿在案解的 L2 段逐步打印它的 distance。
"""
from __future__ import annotations

import json

import numpy as np

from harness import hypo
from harness.env import Action, Game, action_space
from harness.percept import discover
from harness.run import learn_goals


def main():
    sol = json.load(open("solutions.json"))
    seq = [Action.key(i) for i in sol["seq"]]
    game, obs = Game.make("ls20")
    sp = action_space(list(obs.actions) or [1, 2, 3, 4])
    keys = [Action.key(i) for i in sp["keys"]]

    # --- L1: 完全按 agent 的姿势学 ---
    start1 = np.array(obs.grid)
    game.detect_lag(keys)
    ents1, _ = discover(lambda a: np.array(game.effect(a).grid), start1, keys)

    frames = [start1]
    i = 0
    while obs.level == 0:
        obs = game.act(seq[i]); i += 1
        frames.append(np.array(obs.grid))
    # fit 语义: after = 赢步之前那帧 (replay_frames 不含最后一步)
    samples = [hypo.Transition(before=frames[0], after=frames[-2], level=0, ents=ents1)]
    goals = learn_goals(samples, [frames[:-1]])
    print(f"L1 学到 {len(goals)} 条目标:")
    for h, note in goals:
        print(f"  {h.describe()}  [{note}]")
    rel = [h for h, _ in goals if hypo.is_relational(h)]
    if not rel:
        print("没有关系型目标, 停")
        return
    print(f"\n跟踪全部 {len(rel)} 条关系型目标")

    # --- L2: 沿真解逐步打印每条关系型目标的 h ---
    l2_start = np.array(obs.grid)
    print("L2 开局: " + " | ".join(f"{g.describe()} h={g.distance(l2_start):.0f}" for g in rel))
    step = 0
    hs = {g.describe(): [] for g in rel}
    while obs.level == 1 and i < len(seq):
        obs = game.act(seq[i]); i += 1
        step += 1
        g_now = np.array(obs.grid)
        vals = [g.distance(g_now) for g in rel]
        for g, v in zip(rel, vals):
            hs[g.describe()].append(v)
        print(f"  步{step:3d} {seq[i-1]} " + " ".join(f"h={v:.0f}" for v in vals))
    print(f"\nL2 共 {step} 步过关")
    for k, v in hs.items():
        loc = sum(1 for x in v if x < 999)
        print(f"  {k}: 可定位 {loc}/{len(v)} 步, 首={v[0]:.0f} 尾={v[-1]:.0f} 最小={min(v):.0f}")


if __name__ == "__main__":
    main()
