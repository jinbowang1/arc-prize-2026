"""抽象模型层的覆盖率体检:哪些动作建得成常量覆盖表,哪些建不成。

cd82 是标尺(已知 A5 = 50~67 格的大笔、点答案区画 3×4 小块);
r11l 是待测 —— 它每次点击重绘 24~152 格,到底是固定图案还是与状态有关,
这个体检直接决定抽象层能不能用。

**这个脚本只回答"机制是什么",不试图通关。** L7 那次的教训:一个只问机制的
诊断跑,比多搜十层都值钱。
"""
import sys

import numpy as np

from harness.env import Action, Game, action_space
from harness.model import coverage, learn
from harness.percept import analyze
from harness.probe import probe_counters, probe_volatile

for gid in (sys.argv[1:] or ["cd82", "r11l"]):
    game, obs = Game.make(gid)
    sp = action_space(list(obs.actions))
    scene = analyze(obs.grid)
    clicks = [Action.click(c, r) for (r, c) in scene.targets]
    acts = [Action.key(i) for i in sp["keys"]] + clicks
    single = probe_volatile(game, obs, sp["keys"], clicks)
    stepwise = probe_counters(game, obs, acts)
    mask = single & stepwise
    print(f"\n=== {gid} L1 ===")
    print(f"计数器: 单步判据掩 {int((~single).sum())} 格, 跨步判据掩 "
          f"{int((~stepwise).sum())} 格, 合计 {int((~mask).sum())} 格 "
          f"{np.argwhere(~mask).tolist()[:12]}")

    models = learn(game, obs, acts, n_states=4, mask=mask)
    print(coverage(models))
    for m in sorted(models.values(), key=lambda m: -len(m.patch))[:10]:
        print("  " + m.line())
