"""两种分割都取之后, ObjectToObject 仍生不出来。钥匙块在不在? 卡在哪个条件?"""
import json
import numpy as np
from harness.env import Action, Game
from harness.percept import background, by_color, by_figure

seq = json.load(open("solutions.json"))["seq"]
game, obs = Game.make("ls20")
before = None
for t in seq:
    prev = np.array(obs.grid)
    obs = game.act(Action.key(int(t)))
    if obs.level > 0:
        before = prev
        break
after = np.array(obs.grid)
bg, bg_a = background(before), background(after)

def blobs(g, b):
    out = list(by_figure(g, b)) + list(by_color(g, b))
    seen, uniq = set(), []
    for x in out:
        if x.bbox not in seen:
            seen.add(x.bbox); uniq.append(x)
    return uniq

bb, ab = blobs(before, bg), blobs(after, bg_a)
print(f"before {len(bb)} 块 / after {len(ab)} 块 (两种分割合并去重后)")
print("\nbefore 的小块(<=100 格):")
for b in bb:
    n = int(b.mask.sum()) if hasattr(b, "mask") else -1
    r0, r1, c0, c1 = b.bbox
    if (r1-r0+1) * (c1-c0+1) <= 100:
        print(f"   bbox={b.bbox} 尺寸 {r1-r0+1}x{c1-c0+1} center={b.center}")
print("\nafter 的小块(<=100 格):")
for a in ab:
    r0, r1, c0, c1 = a.bbox
    if (r1-r0+1) * (c1-c0+1) <= 100:
        print(f"   bbox={a.bbox} 尺寸 {r1-r0+1}x{c1-c0+1} center={a.center}")

# 逐条件计数
same_shape = moved = overlap = 0
for b in bb:
    kb = b.mask_key(before, bg)
    for a in ab:
        if a.mask_key(after, bg_a) != kb:
            continue
        same_shape += 1
        if a.center == b.center:
            continue
        moved += 1
        ar0, ar1, ac0, ac1 = a.bbox
        for other in ab:
            if other.mask_key(after, bg_a) == kb:
                continue
            orr0, orr1, oc0, oc1 = other.bbox
            if min(ar1, orr1) >= max(ar0, orr0) and min(ac1, oc1) >= max(ac0, oc0):
                overlap += 1
print(f"\n同形状块对 {same_shape} | 其中移动过 {moved} | 其中与他块 bbox 重叠 {overlap}")
