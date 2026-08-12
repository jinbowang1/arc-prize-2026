"""因果目标识别的地面真值检验。

cd82 的结构是已知的(notes/cd82-blindtest.md):
    答案区 (34-43, 27-36) 10×10,题面 (3-12, 3-12) 10×10
如果"动作能改的是答案区、改不了却有内容的是题面"这条因果判据成立,
propose_prompt_answer 应该在 cd82 上直接提出 RegionMatch((34,43,27,36),(3,12,3,12))。

这是唯一算数的检验方式 —— 在一个我已经知道答案的游戏上,看它自己能不能
找出来。r11l 上顺带看它提了什么(那关的答案我不知道,只能存疑)。
"""
import json
import sys

import numpy as np

from harness.env import Action, Game, action_space
from harness.hypo import propose_prompt_answer
from harness.model import collect_states
from harness.percept import analyze, background, discover, mutable_over_states
from harness.probe import probe_counters, probe_volatile
from harness.run import _parse
from harness.search import bfs_level_up

TRUTH = {"cd82": ((34, 43, 27, 36), (3, 12, 3, 12))}


def goto(gid, upto):
    game, obs = Game.make(gid)
    try:
        sol = json.load(open(f"{gid}_solutions.json"))
        seq = [_parse(s) for s in sol["seq"]]
        for a in seq[:sum(sol["per_level_steps"][:upto])]:
            obs = game.act(a)
        return game, obs
    except FileNotFoundError:
        pass
    while obs.level < upto:
        sp = action_space(list(obs.actions))
        scene = analyze(obs.grid)
        clicks = [Action.click(c, r) for (r, c) in scene.targets]
        mask = probe_volatile(game, obs, sp["keys"], clicks) & probe_counters(game, obs, clicks)
        res = bfs_level_up(game, obs, sp["keys"], 6 if sp["clicks"] else None, mask,
                           max_depth=20, max_nodes=20000, max_seconds=60)
        if not res.solved:
            raise SystemExit(f"走不到 L{upto+1}")
        for a in res.seq:
            obs = game.act(a)
    return game, obs


for gid, lv in [("cd82", 0), ("cd82", 2), ("r11l", 1)]:
    if len(sys.argv) > 1 and gid != sys.argv[1]:
        continue
    game, obs = goto(gid, lv)
    sp = action_space(list(obs.actions))
    scene = analyze(obs.grid)
    clicks = [Action.click(c, r) for (r, c) in scene.targets]
    acts = [Action.key(i) for i in sp["keys"]] + clicks
    hud = probe_volatile(game, obs, sp["keys"], clicks) & probe_counters(game, obs, acts)

    g = np.array(obs.grid)
    ents, mut1 = discover(lambda a: np.array(game.peek(a).grid), g, acts)
    states = collect_states(game, obs, acts, 5)
    mutable = mutable_over_states([lambda a, c=c: np.array(c.peek(a).grid) for c, _ in states],
                                  [np.array(o.grid) for _, o in states], acts)
    mutable &= hud                      # 计数器不算答案区
    print(f"\n=== {gid} L{lv+1} ===")
    print(f"可变格 单状态 {int((mut1 & hud).sum())} -> 多状态并集 {int(mutable.sum())} / 4096, "
          f"实体 {len(ents)} 个")

    props = propose_prompt_answer(g, mutable, background(g))
    print(f"因果目标假设 {len(props)} 条:")
    for p in props[:6]:
        d = p.distance(g)
        print(f"  {p.describe()}  开局 h={d:.0f}")
    if gid in TRUTH:
        want_a, want_t = TRUTH[gid]
        hit = [p for p in props if p.a == want_a and p.b == want_t]
        near = [p for p in props if p.a == want_a]
        print(f"  地面真值 答案区{want_a} 题面{want_t}: "
              f"{'✅命中' if hit else ('答案区对但题面不对' if near else '❌没提出来')}")
