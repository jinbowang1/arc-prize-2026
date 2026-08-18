"""L3: 如果语义穷尽仍无解, 用它找第三个维度。

穷尽实验的指纹是 (9格二值, 方块位置)。若穷尽无解, 说明过关条件不在这两维里,
必然还有个我没建模的东西在变。

做法不是猜, 是**全屏帧 diff**: 走一批随机但确定的动作序列, 记录每一帧,
统计**哪些格子变过**, 再扣掉已知的三块(画布/方块可达区/已掩的计数器)。
剩下的就是第三维度的候选。

⚠️与"实体发现"的判据一致: 总是一起变的格子归为同一实体。
⚠️用真 noop 补步, 不污染任何维度。
"""
from __future__ import annotations

import json

import numpy as np

from harness.env import Action, Game, action_space
from harness.percept import analyze
from harness.run import _parse

sol = json.load(open("sc25_solutions.json"))
game, obs = Game.make("sc25")
for t in sol["seq"]:
    obs = game.act(_parse(t))
g0 = np.array(obs.grid)
sp = action_space(list(obs.actions))
sc = analyze(obs.grid)
clicks = [Action.click(c, r) for (r, c) in sc.targets]
acts = [Action.key(k) for k in sp["keys"]] + clicks
game.detect_lag(acts)

R, C = [49, 54, 59], [24, 29, 34]
KNOWN = np.zeros((64, 64), dtype=bool)
KNOWN[47:64, 22:39] = True        # 画布(九宫格)
KNOWN[10:34, 26:46] = True        # 方块可达的管道区
KNOWN[0:6, 60:64] = True          # 右上角计数器

# 走 N 条确定性序列, 累积"变过"的格子
changed = np.zeros((64, 64), dtype=bool)
for seed in range(12):
    n = game.fork()
    cur = g0
    for t in range(14):
        a = acts[(seed * 7 + t * 3) % len(acts)]
        o = n.act(a)
        if o.dead:
            break
        g = np.array(o.grid)
        changed |= (g != cur)
        cur = g

extra = changed & ~KNOWN
print(f"12 条序列 x 14 步: 变过的格 {int(changed.sum())}, "
      f"其中**已知三块之外** {int(extra.sum())} 格")
if extra.any():
    cells = np.argwhere(extra)
    rows = sorted({int(r) for r, _ in cells})
    print(f"  行范围 {rows[0]}..{rows[-1]}")
    # 按行分组打印
    for r in rows:
        cs = sorted(int(c) for rr, c in cells if rr == r)
        print(f"    行{r:>3}: 列 {cs}")
else:
    print("  => 已知三块之外**没有任何格子变过**;"
          "\n     第三维度不是画面上的东西(可能是不可见的内部计数/顺序)。")
