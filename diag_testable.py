"""诊断: 开局判不了的那 35 格是哪些格? 铺底为什么一格也没提高?

判 65/100 在换了三种造底方式之后**纹丝不动**。按本项目反复应验的判据,
这时候该问机制, 不该再调采集参数。

问三件事:
  1. 能判的 65 格 / 判不了的 35 格, 各自在答案区的什么位置
  2. 卡住的那 12 格(局部坐标 行0-2 列3-6)在哪一边
  3. 轨迹底到底收到了几个底、底之间差异多大 —— fallback 是没触发, 还是触发了没用
"""
from __future__ import annotations

import json
import time

import numpy as np

from harness.canvas import (_config_fp, _region, _trajectory_floors, classify,
                            collect_brushes)
from harness.env import Action, Game, action_space
from harness.hypo import propose_prompt_answer
from harness.model import collect_states
from harness.percept import analyze, mutable_over_states
from harness.probe import run_probe
from harness.run import _parse

GID, LV = "cd82", 2
sol = json.load(open(f"{GID}_solutions.json"))
game, obs = Game.make(GID)
for a in [_parse(s) for s in sol["seq"]][:sum(sol["per_level_steps"][:LV])]:
    obs = game.act(a)

t0 = time.time()
sp = action_space(list(obs.actions))
scene = analyze(obs.grid)
clicks = [Action.click(c, r) for (r, c) in scene.targets]
acts = [Action.key(i) for i in sp["keys"]] + clicks
rep = run_probe(game, obs, sp["kind"], sp["keys"], clicks)
states = collect_states(game, obs, acts, 5)
mut = mutable_over_states([lambda a, c=c: np.array(c.peek(a).grid) for c, _ in states],
                          [np.array(o.grid) for _, o in states], acts) & rep.mask
h = propose_prompt_answer(np.array(obs.grid), mut, scene.bg)[0]
BOX = h.a
target = _region(np.array(obs.grid), h.b)
start = _region(np.array(obs.grid), BOX)
st = classify(game, obs, acts, BOX)

# ---- 复刻 collect_brushes 的第一档造底, 把 testable 摊出来 ----
base_cfg = _config_fp(np.array(obs.grid), BOX, rep.mask)
floors = [(game.fork(), _region(np.array(obs.grid), BOX))]
for sub in st.submitters[:4]:
    f = game.fork()
    o = f.act(sub)
    if o.dead or o.level != obs.level:
        continue
    if _config_fp(np.array(o.grid), BOX, rep.mask) != base_cfg:
        continue
    floors.append((f, _region(np.array(o.grid), BOX)))
bases = [r for _, r in floors]
testable = np.zeros_like(bases[0], dtype=bool)
for i in range(len(bases)):
    for j in range(i + 1, len(bases)):
        testable |= (bases[i] != bases[j])
print(f"第一档: {len(floors)} 个底, 可判 {int(testable.sum())}/{testable.size} 格", flush=True)

print("\n[1] 可判性地图 (o=可判, .=判不了; 红色数字 = 最后卡住的那 12 格)", flush=True)
for r in range(testable.shape[0]):
    row = ""
    for c in range(testable.shape[1]):
        ch = "o" if testable[r, c] else "."
        if 0 <= r <= 2 and 3 <= c <= 6:
            ch = f"\033[31m{ch}\033[0m"
        row += ch + " "
    print("    " + row, flush=True)

stuck = np.zeros_like(testable)
stuck[0:3, 3:7] = True
# ⚠️判据必须只在 stuck 的格子上求值。第一版写成 (testable & stuck).all(),
# 那是对全部 100 格求值 —— 非 stuck 位置恒为 False, 结论永远是 ❌, 和上面
# 打印的 12/12 自相矛盾。数字对、结论反, 比两个都错更容易骗过人。
print(f"\n[2] 卡住的 12 格里, 可判的有 {int((testable & stuck).sum())}/12 格"
      f" -> {'✅ 它们能判, 问题不在这' if testable[stuck].all() else '❌ **它们正好在判不了的那半边**'}",
      flush=True)

print(f"\n[各底之间的差异]", flush=True)
for i in range(len(bases)):
    for j in range(i + 1, len(bases)):
        d = (bases[i] != bases[j])
        print(f"    底{i} vs 底{j}: 差 {int(d.sum())} 格", flush=True)

# ---- 轨迹底: 到底收到几个 ----
print(f"\n[3] 轨迹底 fallback", flush=True)
cands = [[a] for a in st.adjusters[:3]]
cands += [[a, b] for a in st.adjusters[:2] for b in st.adjusters[1:3] if repr(a) != repr(b)]
for anchor in cands:
    root, robs, tf = _trajectory_floors(game, obs, st, anchor, BOX, rep.mask)
    if root is None:
        print(f"    anchor={[str(a) for a in anchor]}: ❌ 走不通", flush=True)
        continue
    tb = [r for _, r in tf]
    t2 = np.zeros_like(tb[0], dtype=bool)
    for i in range(len(tb)):
        for j in range(i + 1, len(tb)):
            t2 |= (tb[i] != tb[j])
    print(f"    anchor={[str(a) for a in anchor]}: {len(tf)} 个底, 可判 {int(t2.sum())}/100 格"
          f"{'  <- 比第一档好' if int(t2.sum()) > int(testable.sum()) else ''}", flush=True)

# ---- 那 35 格到底能不能被任何一支笔改变? ----
print(f"\n[4] 判不了的格子, 有没有任何动作能改变它们", flush=True)
never = np.zeros_like(testable)
g0 = _region(np.array(obs.grid), BOX)
changed_any = np.zeros_like(testable)
node = game.fork()
seen = set()
q = [(game.fork(), obs, 0)]
tried = 0
while q and tried < 60:
    nd, ob, d = q.pop(0)
    for sub in st.submitters:
        o = nd.fork().act(sub)
        if o.dead or o.level != obs.level:
            continue
        changed_any |= (_region(np.array(o.grid), BOX) != _region(np.array(ob.grid), BOX))
        tried += 1
    if d < 3:
        for a in st.adjusters[:6]:
            ch = nd.fork()
            o = ch.act(a)
            if o.dead or o.level != obs.level:
                continue
            fp = _config_fp(np.array(o.grid), BOX, rep.mask)
            if fp in seen:
                continue
            seen.add(fp)
            q.append((ch, o, d + 1))
print(f"    在 {tried} 次提交里被改变过的格子: {int(changed_any.sum())}/100", flush=True)
print(f"    判不了但**被改变过**的格子: {int((~testable & changed_any).sum())} "
      f"-> 这些是采集漏掉的", flush=True)
print(f"    判不了且**从没被改变过**的格子: {int((~testable & ~changed_any).sum())} "
      f"-> 这些是笔够不着的", flush=True)
print(f"    卡住的 12 格被改变过的: {int((changed_any & stuck).sum())}/12", flush=True)
print(f"\n总耗时 {time.time()-t0:.0f}s", flush=True)
