"""分层渲染模型的验收:在卡住的那一关上,它预测得准不准?

判据是**留出组合上的整帧一致率**,不是逐格一致率 —— 画面绝大部分是背景,
什么都不预测也有 95% 的逐格一致。

靶子:cd82 L3(已知卡点)和 r11l L2(盲测卡点)。
"""
import json
import sys

import numpy as np

from harness.env import Action, Game, action_space
from harness.factored import learn_slots
from harness.layers import learn_layers, plan_on_layers
from harness.percept import analyze
from harness.probe import probe_counters, probe_volatile
from harness.run import _parse
from harness.search import bfs_level_up


def goto(gid, upto):
    """走到第 upto 关(0-indexed)。有解就照解走,没有就 BFS。"""
    game, obs = Game.make(gid)
    try:
        sol = json.load(open(f"{gid}_solutions.json"))
        seq = [_parse(s) for s in sol["seq"]]
        n = sum(sol["per_level_steps"][:upto])
        for a in seq[:n]:
            obs = game.act(a)
        return game, obs
    except FileNotFoundError:
        pass
    while obs.level < upto:
        scene = analyze(obs.grid)
        clicks = [Action.click(c, r) for (r, c) in scene.targets]
        sp = action_space(list(obs.actions))
        mask = probe_volatile(game, obs, sp["keys"], clicks) & probe_counters(game, obs, clicks)
        res = bfs_level_up(game, obs, sp["keys"], 6 if sp["clicks"] else None, mask,
                           max_depth=20, max_nodes=20000, max_seconds=60)
        if not res.solved:
            raise SystemExit(f"走不到 L{upto+1}")
        for a in res.seq:
            obs = game.act(a)
    return game, obs


for gid, lv in [("cd82", 2), ("r11l", 1)]:
    if len(sys.argv) > 1 and gid != sys.argv[1]:
        continue
    game, obs = goto(gid, lv)
    sp = action_space(list(obs.actions))
    scene = analyze(obs.grid)
    clicks = [Action.click(c, r) for (r, c) in scene.targets]
    acts = [Action.key(i) for i in sp["keys"]] + clicks
    mask = probe_volatile(game, obs, sp["keys"], clicks) & probe_counters(game, obs, acts)

    print(f"\n=== {gid} L{lv+1} ===", flush=True)
    sm = learn_slots(game, obs, acts, mask)
    print(sm.text(), flush=True)
    lm = learn_layers(game, obs, sm, mask)
    print(lm.text(), flush=True)

    if lm.usable:
        # 拿一个有梯度的量试试规划:与题面的差异格数
        g0 = np.array(obs.grid)
        p = plan_on_layers(lm, sm, lambda g: float(((g != g0) & mask).sum()))
        print("  " + p.text(), flush=True)
