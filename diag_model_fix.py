"""验证 model.learn 改用 effect 语义后, 各局的"可用/无效果"数字变了没。
这三个数字整晚没动过(可用 0 / 状态相关 N / 无效果 M), 是抽象层建不起来的直接原因。"""
import sys
import numpy as np
from harness import model
from harness.env import Action, Game, action_space
from harness.percept import analyze
from harness.probe import run_probe

for gid in (sys.argv[1:] or ["sc25", "cd82", "r11l"]):
    game, obs = Game.make(gid)
    sp = action_space(list(obs.actions))
    sc = analyze(obs.grid)
    clicks = [Action.click(c, r) for (r, c) in sc.targets]
    acts = [Action.key(i) for i in sp["keys"]] + clicks
    lag = game.detect_lag(acts)
    rep = run_probe(game, obs, sp["kind"], sp["keys"], clicks)
    ms = model.learn(game, obs, acts, mask=rep.mask)
    usable = sum(1 for m in ms.values() if m.patch and not m.conflicts)
    stateful = sum(1 for m in ms.values() if m.conflicts)
    noeffect = sum(1 for m in ms.values() if not m.patch and not m.conflicts and not m.kills)
    best = max((len(m.patch) for m in ms.values()), default=0)
    print(f"{gid:6} lagged={str(lag):5} | 动作 {len(acts):>3} | "
          f"**可用 {usable}**(最大覆盖 {best} 格) / 状态相关 {stateful} / 无效果 {noeffect}")
