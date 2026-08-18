"""放宽判据后 fit 仍无 ObjectToObject: 是候选没生成, 还是生成了被过滤?
fit 末尾有"必须条条样本 after 成立、before 不成立"的过滤。"""
import json
import numpy as np
from harness import hypo
from harness.env import Action, Game
from harness.percept import background, by_figure

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
bb, ab = by_figure(before, bg), by_figure(after, bg_a)

print("before 各块:", [(b.bbox, b.mask_key(before, bg)[:8]) for b in bb])
print("after  各块:", [(a.bbox, a.mask_key(after, bg_a)[:8]) for a in ab])

# 复现**新**逻辑, 数候选
gen = []
for b in bb:
    kb = b.mask_key(before, bg)
    for a in ab:
        if a.mask_key(after, bg_a) != kb or a.center == b.center:
            continue
        ar0, ar1, ac0, ac1 = a.bbox
        for other in ab:
            ko = other.mask_key(after, bg_a)
            if ko == kb:
                continue
            orr0, orr1, oc0, oc1 = other.bbox
            if min(ar1, orr1) >= max(ar0, orr0) and min(ac1, oc1) >= max(ac0, oc0):
                gen.append(("ObjectToObject", kb[:6], ko[:6]))
print(f"\n新逻辑生成 ObjectToObject 候选: {len(gen)} 条 {gen[:4]}")

# 再看过滤: 候选要 after 成立 & before 不成立
s = hypo.Transition(before=before, after=after, level=0)
for kb_, ko_ in [(g[1], g[2]) for g in gen[:3]]:
    pass
allc = []
for b in bb:
    kb = b.mask_key(before, bg)
    for a in ab:
        if a.mask_key(after, bg_a) != kb or a.center == b.center:
            continue
        ar0, ar1, ac0, ac1 = a.bbox
        for other in ab:
            ko = other.mask_key(after, bg_a)
            if ko == kb: continue
            orr0, orr1, oc0, oc1 = other.bbox
            if min(ar1, orr1) >= max(ar0, orr0) and min(ac1, oc1) >= max(ac0, oc0):
                allc.append(hypo.ObjectToObject(kb, ko))
print(f"\n过滤检查(after 成立 & before 不成立):")
for h in allc[:5]:
    print(f"   {h.describe()[:50]}  after={h.is_goal(after)} before={h.is_goal(before)}")
