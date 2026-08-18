"""ls20 L1 通关瞬间, fit 到底拟合出了什么? ObjectToObject 差在哪一步?

已知: L1 通关后只沉淀出 3 条 color_count(绝对型, 参数是上一关常数),
ObjectToObject(钥匙匹配锁 —— ls20 的真判据)一条没有。
"""
import json
import numpy as np
from harness import hypo
from harness.env import Action, Game
from harness.percept import background, by_figure

# 走 L1 的在案解, 取通关前后两帧
seq = json.load(open("solutions.json"))["seq"]
game, obs = Game.make("ls20")
before = None
for t in seq:
    a = Action.key(int(t))
    prev = np.array(obs.grid)
    obs = game.act(a)
    if obs.level > 0:
        before = prev
        break
after = np.array(obs.grid)
print(f"通关前后两帧取到, before/after shape {before.shape}")

s = hypo.Transition(before=before, after=after, level=0)
bg, bg_a = background(before), background(after)
bb, ab = by_figure(before, bg), by_figure(after, bg_a)
print(f"before 块 {len(bb)} 个, after 块 {len(ab)} 个\n")

# 复现 fit 里 ObjectToObject 那段, 看卡在哪
n_same_shape = n_moved = n_concentric = 0
for b in bb:
    kb = b.mask_key(before, bg)
    for a in ab:
        if a.mask_key(after, bg_a) != kb:
            continue
        n_same_shape += 1
        if a.center == b.center:
            continue
        n_moved += 1
        for other in ab:
            if other.mask_key(after, bg_a) != kb and other.center == a.center:
                n_concentric += 1
        break
print(f"同形状的块对: {n_same_shape}")
print(f"其中位置变了的: {n_moved}")
print(f"其中 after 里有**同心**的另一形状: {n_concentric}  <- ObjectToObject 要靠它")

got = hypo.fit([s])
print(f"\nfit 实际产出 {len(got)} 条:")
for h in got[:8]:
    print(f"   {type(h).__name__:18} {h.describe()[:60]}")
