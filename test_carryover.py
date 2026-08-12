"""关卡交接单的实测:拿已通关的解走关,看每关开始时 diff 出什么。

cd82 有完整六关的解(cd82_solutions.json),是最好的样本 —— 它逐关加颜色数
(L1 两色 / L2 三色 / L3 四色),交接单应该把这件事直接说出来。
"""
import json
import sys

import numpy as np

from harness.carryover import Knowledge, brief, shape_keys
from harness.env import Action, Game
from harness.percept import analyze
from harness.run import _parse

GID = sys.argv[1] if len(sys.argv) > 1 else "cd82"
sol = json.load(open(f"{GID}_solutions.json"))
seq = [_parse(s) for s in sol["seq"]]
per = sol["per_level_steps"]

game, obs = Game.make(GID)
know = Knowledge()
prev_scene = None
i = 0
for lv, n in enumerate(per):
    scene = analyze(obs.grid)
    b = brief(lv, obs, scene, know if lv else None, prev_scene)
    print(b.text(), flush=True)

    g = np.array(obs.grid)
    know.colors |= set(int(x) for x in np.unique(g))
    know.shapes |= shape_keys(g)
    know.learned_at = lv
    prev_scene = scene

    for a in seq[i:i + n]:
        obs = game.act(a)
    i += n
    print(f"        (走完 {n} 步 -> level {obs.level})\n", flush=True)
