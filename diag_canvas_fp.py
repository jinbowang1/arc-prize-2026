"""诊断: 构型指纹里混进了什么?

上一个诊断测出「落 3 笔之后的构型图与开局重合 0 个」。构型指纹 = 挖掉答案区
之后的整帧, 理论上只该装「画笔构型」。重合 0 个说明画面上还有别的东西在跟着
画布一起变 —— 指纹被污染, BFS 会把同一个真实构型拆成好几个。

这个脚本直接问: **同一条调整序列, 在两个不同画布上走完, 画面差在哪几个格。**
差异格如果集中成一小片, 那就是没被 mask 掉的计数器/显示器(cd82 上已经栽过一次:
"多出的 (63,59)~(63,63) 是之前完全没发现的计数器")。
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from harness.canvas import _config_fp, _region, classify
from harness.env import Action, Game, Obs, action_space
from harness.hypo import propose_prompt_answer
from harness.model import collect_states
from harness.percept import analyze, mutable_over_states
from harness.probe import run_probe
from harness.run import _parse

GID = sys.argv[1] if len(sys.argv) > 1 else "cd82"
LV = int(sys.argv[2]) if len(sys.argv) > 2 else 2

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
st = classify(game, obs, acts, BOX)
print(f"答案区 {BOX} | 提交 {len(st.submitters)} 调整 {len(st.adjusters)}", flush=True)
print(f"probe 掩掉的格数 {int((~rep.mask).sum())}: {np.argwhere(~rep.mask)[:12].tolist()}",
      flush=True)


def walk(g: Game, o: Obs, seq):
    node = g.fork()
    cur = o
    for a in seq:
        cur = node.act(a)
        if cur.dead:
            return None, None
    return node, cur


# 造两个只有画布不同的状态: 一个是原样, 一个是按一次提交(提交不改构型, 已验 60/60)
subA = st.submitters[0]
nB, oB = walk(game, obs, [subA])
if nB is None:
    raise SystemExit("提交动作走不通")
print(f"两个底: 答案区差 {int((_region(np.array(obs.grid), BOX) != _region(np.array(oB.grid), BOX)).sum())} 格",
      flush=True)

gA, gB = np.array(obs.grid), np.array(oB.grid)
d = (gA != gB)
d_out = d.copy()
r0, r1, c0, c1 = BOX
d_out[r0:r1 + 1, c0:c1 + 1] = False          # 答案区内的差异是应该的
print(f"[基线] 答案区**外**的差异 {int(d_out.sum())} 格: {np.argwhere(d_out)[:16].tolist()}",
      flush=True)
print(f"       其中被 mask 掩掉的 {int((d_out & ~rep.mask).sum())} 格, "
      f"**没被掩掉的 {int((d_out & rep.mask).sum())} 格** <- 这些就是污染源", flush=True)

# 再走一条相同的调整序列, 看污染是否跟着传播
adj = st.adjusters[:6]
for k in range(1, 4):
    seq = adj[:k]
    nA2, oA2 = walk(game, obs, seq)
    nB2, oB2 = walk(nB, oB, seq)
    if nA2 is None or nB2 is None:
        continue
    fa = _config_fp(np.array(oA2.grid), BOX, rep.mask)
    fb = _config_fp(np.array(oB2.grid), BOX, rep.mask)
    dd = (np.array(oA2.grid) != np.array(oB2.grid))
    dd[r0:r1 + 1, c0:c1 + 1] = False
    bad = np.argwhere(dd & rep.mask)
    print(f"[走 {k} 步调整] 构型指纹{'相同 ✅' if fa == fb else '不同 ❌'} | "
          f"答案区外未掩差异 {len(bad)} 格: {bad[:10].tolist()}", flush=True)

# ---- 落笔之后, 构型图为什么对不上 --------------------------------------
# 指纹本身没被污染(上面已验)。那 "重合 0 个" 只剩两种解释:
#   ① 构型图不强连通 —— 走出去就回不来(动作单向)
#   ② 落笔改变了答案区之外的画面 —— 那才是真的污染
# 直接比对落笔前后的整帧, 一次问清楚。
from harness.canvas import collect_brushes, plan_canvas  # noqa: E402

brushes, complete, judged, total, ncfg = collect_brushes(
    game, obs, st, rep.mask, max_configs=80, max_seconds=120)
start = _region(np.array(obs.grid), BOX)
target = _region(np.array(obs.grid), h.b)
plan = plan_canvas(start, target, brushes)
print(f"\n[落笔诊断] 抽象层 {len(plan.brushes)} 笔, 画笔 {len(brushes)} 支", flush=True)

node, cur, laid = game.fork(), obs, 0
for b in plan.brushes[:3]:
    n2, c2 = walk(node, cur, b.seq)
    if n2 is None:
        break
    c3 = n2.act(b.submit)
    if c3.dead or c3.level != obs.level:
        break
    node, cur, laid = n2, c3, laid + 1
    dd = (np.array(cur.grid) != gA)
    dd[r0:r1 + 1, c0:c1 + 1] = False
    out_cells = np.argwhere(dd & rep.mask)
    print(f"  落第 {laid} 笔后: 答案区外未掩差异 {len(out_cells)} 格 "
          f"{out_cells[:12].tolist()}", flush=True)

# 从落笔后的构型往回走: 能不能回到开局构型?
fp0 = _config_fp(gA, BOX, rep.mask)
fpN = _config_fp(np.array(cur.grid), BOX, rep.mask)
print(f"  开局构型 == 落 {laid} 笔后构型 ? {fp0 == fpN}", flush=True)

seen = {fpN}
frontier = [(node, cur)]
found = False
for depth in range(1, 5):
    nxt = []
    for nd, ob in frontier[:40]:
        for a in st.adjusters:
            ch = nd.fork()
            o2 = ch.act(a)
            if o2.dead or o2.level != obs.level:
                continue
            f = _config_fp(np.array(o2.grid), BOX, rep.mask)
            if f == fp0:
                found = True
                break
            if f in seen:
                continue
            seen.add(f)
            nxt.append((ch, o2))
        if found:
            break
    print(f"  回溯深度 {depth}: 见过 {len(seen)} 个构型{'  ✅ 走回开局构型了' if found else ''}",
          flush=True)
    if found:
        break
    frontier = nxt
if not found:
    print("  ❌ 4 层内走不回开局构型 —— 构型图**不强连通**, "
          "开局采的画笔库在落笔后根本不适用", flush=True)

print(f"总耗时 {time.time()-t0:.0f}s", flush=True)
