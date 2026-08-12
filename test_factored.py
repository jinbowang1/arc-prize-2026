"""分槽 + 槽搜索,在 r11l L2 上试。

对照组是真机 BFS:同一关跑 180 秒,最深 6 层,h 从 7 降到 2,未解出。
"""
import sys

import numpy as np

from harness.env import Action, Game, action_space
from harness.factored import learn_slots, slot_search
from harness.percept import analyze
from harness.probe import probe_counters, probe_volatile
from harness.search import bfs_level_up

GID = sys.argv[1] if len(sys.argv) > 1 else "r11l"
UPTO = int(sys.argv[2]) if len(sys.argv) > 2 else 6

game, obs = Game.make(GID)
sp = action_space(list(obs.actions))
solution = []

while obs.level < min(UPTO, obs.win_levels):
    lv = obs.level
    scene = analyze(obs.grid)
    clicks = [Action.click(c, r) for (r, c) in scene.targets]
    acts = [Action.key(i) for i in sp["keys"]] + clicks
    mask = probe_volatile(game, obs, sp["keys"], clicks) & probe_counters(game, obs, acts)

    res = bfs_level_up(game, obs, sp["keys"], 6 if sp["clicks"] else None, mask,
                       max_depth=20, max_nodes=20000, max_seconds=45)
    if res.solved:
        print(f"L{lv+1}: BFS {res.text()}", flush=True)
        seq = res.seq
    else:
        print(f"L{lv+1}: BFS 交白卷 -> 转分槽", flush=True)
        sm = learn_slots(game, obs, acts, mask)
        print("  " + sm.text(), flush=True)
        sr = slot_search(game, obs, sm, mask=mask, max_seconds=180)
        print("  " + sr.text(), flush=True)
        if not sr.solved:
            break
        seq = sr.seq
    for a in seq:
        obs = game.act(a)
    solution.append([str(a) for a in seq])
    print(f"  -> 到 level {obs.level}, 本关 {len(seq)} 步", flush=True)

print("\n解:", solution)
