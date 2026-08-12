"""实体发现的回归测试。

判据来自两个真实 case:
  - cd82 L1: 画面上有**两个面板**(大 12×7 + 小 4×3), 归不同动作管。
    旧版按"变化区重叠就合并"能不能分开是偶然的; 新版按"总是一起变的格子"
    应该稳定分出两组。这正是 cd82 L4 卡几小时的那个漏掉的实体。
  - r11l L1: 纯 click 重绘型游戏, 旧版并成 1 个覆盖整屏的"实体"(396 格),
    等于什么都没发现。新版至少要分出多个。
"""
import numpy as np

from harness.env import Action, Game, action_space
from harness.percept import analyze, discover_entities

for gid in ("cd82", "r11l"):
    game, obs = Game.make(gid)
    sp = action_space(list(obs.actions))
    scene = analyze(obs.grid)
    acts = ([Action.key(i) for i in sp["keys"]] +
            [Action.click(c, r) for (r, c) in scene.targets])
    ents = discover_entities(lambda a: np.array(game.peek(a).grid),
                             np.array(obs.grid), acts)
    print(f"\n=== {gid} L1: 动作 {len(acts)} 个 -> 实体 {len(ents)} 个 ===")
    for e in ents[:8]:
        h, w = e.bbox[1] - e.bbox[0] + 1, e.bbox[3] - e.bbox[2] + 1
        print(f"  {h}x{w} @ {e.bbox} {e.cells}格 受{len(e.movers)}个动作影响 {e.movers[:3]}")
