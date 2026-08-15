"""诊断: 轨迹底为什么没救回来?

改完 collect_brushes 加了第三种造底, cd82 L3 第 3 笔仍然报判 0/100 —— 数字
**纹丝不动**。按项目自己的判据, 数字一格不动通常说明那条路根本没跑到, 或者
跑到了但前提不成立。这个脚本复现「落 3 笔之后」的状态, 逐条问清楚:

  - anchor 走不走得通
  - 插了提交之后, 终点构型还等不等于 anchor 终点(即"提交不改构型"在**这个
    状态下**还成不成立 —— 我只在开局验过 60/60, 那是"采样只在一个状态上做"
    的老毛病)
  - 收到的底之间到底有没有差异
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from harness.canvas import (_config_fp, _region, _trajectory_floors, classify,
                            collect_brushes, plan_canvas, solve)
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
acts = [Action.key(i) for i in sp["keys"]] + [Action.click(c, r) for (r, c) in scene.targets]
rep = run_probe(game, obs, sp["kind"], sp["keys"], [Action.click(c, r) for (r, c) in scene.targets])
states = collect_states(game, obs, acts, 5)
mut = mutable_over_states([lambda a, c=c: np.array(c.peek(a).grid) for c, _ in states],
                          [np.array(o.grid) for _, o in states], acts) & rep.mask
h = propose_prompt_answer(np.array(obs.grid), mut, scene.bg)[0]
BOX = h.a
target = _region(np.array(obs.grid), h.b)
st = classify(game, obs, acts, BOX)


def acts_fn(o):
    sc = analyze(o.grid)
    return ([Action.key(i) for i in sp["keys"]] +
            [Action.click(c, r) for (r, c) in sc.targets])


# 复现「落 3 笔之后」: 直接让闭环跑 3 笔, 再把它走过的序列在克隆体上重放
seq3, _o3, why3 = solve(game, obs, st, target, rep.mask, max_strokes=3,
                        acts_fn=acts_fn, max_configs=2000, collect_seconds=300)
node = game.fork()
cur = obs
for a in seq3:
    cur = node.act(a)
print(f"复现落 3 笔: {len(seq3)} 步, level={cur.level}, "
      f"离目标差 {int((_region(np.array(cur.grid), BOX) != target).sum())} 格 | {time.time()-t0:.0f}s",
      flush=True)

# 这个状态下的动作二分(跨状态取并集, 和 solve 里一样)
fresh = classify(node, cur, acts_fn(cur), BOX)
subs = {repr(a): a for a in st.submitters}
subs.update({repr(a): a for a in fresh.submitters})
adjs = {repr(a): a for a in st.adjusters}
adjs.update({repr(a): a for a in fresh.adjusters})
for k in subs:
    adjs.pop(k, None)
from harness.canvas import CanvasSetup  # noqa: E402
st3 = CanvasSetup(answer_box=BOX, submitters=list(subs.values()), adjusters=list(adjs.values()))
print(f"此状态动作二分: 提交 {len(st3.submitters)} 调整 {len(st3.adjusters)}", flush=True)

# ---- 逐条问 anchor ------------------------------------------------------
base_cfg_now = _config_fp(np.array(cur.grid), BOX, rep.mask)
print("\n[逐条诊断 anchor]", flush=True)
for a in st3.adjusters[:6]:
    root, robs, floors = _trajectory_floors(game=node, obs=cur, st=st3, anchor=[a],
                                            box=BOX, mask=rep.mask)
    if root is None:
        print(f"  anchor={a}: ❌ anchor 本身走不通(死了或过关了)", flush=True)
        continue
    bases = [reg for _, reg in floors]
    diffs = 0
    for i in range(len(bases)):
        for j in range(i + 1, len(bases)):
            diffs = max(diffs, int((bases[i] != bases[j]).sum()))
    print(f"  anchor={a}: 收到 {len(floors)} 个底, 底间最大差异 {diffs} 格"
          f"{'  <- 有差异, 能判' if diffs > 0 else '  ❌ 底全一样'}", flush=True)

    # 自证失败的有几个? 手工重做一遍统计
    ok = rej = walkfail = 0
    for i in range(2):
        for sub in st3.submitters:
            f = game.fork()
            f = node.fork()
            o = cur
            bad = False
            for act in ([a][:i] + [sub] + [a][i:]):
                o = f.act(act)
                if o.dead or o.level != cur.level:
                    bad = True
                    break
            if bad:
                walkfail += 1
                continue
            if _config_fp(np.array(o.grid), BOX, rep.mask) == _config_fp(
                    np.array(node.fork().act(a).grid), BOX, rep.mask):
                ok += 1
            else:
                rej += 1
    print(f"      候选底: 走通并自证通过 {ok}, **构型自证被拒 {rej}**, 走不通 {walkfail}",
          flush=True)

# ---- 提交动作到底改了构型的哪里 ----------------------------------------
# 骨架("提交不改构型")在开局实测 60/60 成立, 在这里 0/10 成立。
# 差异格的位置就是机制本身 —— 直接打印, 不猜。
print("\n[提交动作改了构型的哪里]", flush=True)
r0, r1, c0, c1 = BOX
g_now = np.array(cur.grid)
for sub in st3.submitters:
    c = node.fork()
    o = c.act(sub)
    if o.dead or o.level != cur.level:
        print(f"  {sub}: 走不通(dead={o.dead}, level={o.level})", flush=True)
        continue
    g2 = np.array(o.grid)
    d = (g_now != g2)
    d_in = d.copy()
    d_out = d.copy()
    d_out[r0:r1 + 1, c0:c1 + 1] = False
    d_in[:] = False
    d_in[r0:r1 + 1, c0:c1 + 1] = d[r0:r1 + 1, c0:c1 + 1]
    cells = np.argwhere(d_out & rep.mask)
    print(f"  {sub}: 答案区内改 {int(d_in.sum())} 格 | 答案区外未掩改 {len(cells)} 格", flush=True)
    if len(cells):
        rows = sorted(set(int(x) for x, _ in cells))
        cols = sorted(set(int(y) for _, y in cells))
        print(f"      行 {rows[:12]} 列 {cols[:12]}", flush=True)
        print(f"      前 12 格 {cells[:12].tolist()}", flush=True)
        # 值怎么变的
        for (rr, cc) in cells[:6]:
            print(f"        ({rr},{cc}): {g_now[rr,cc]} -> {g2[rr,cc]}", flush=True)

# 同一件事在开局问一遍, 做对照 —— 确认差别是状态带来的, 不是我测法变了
print("\n[对照: 同样的问法在关卡开局]", flush=True)
g0 = np.array(obs.grid)
for sub in st.submitters[:5]:
    c = game.fork()
    o = c.act(sub)
    if o.dead:
        continue
    g2 = np.array(o.grid)
    d = (g0 != g2)
    d[r0:r1 + 1, c0:c1 + 1] = False
    cells = np.argwhere(d & rep.mask)
    print(f"  {sub}: 答案区外未掩改 {len(cells)} 格 {cells[:6].tolist()}", flush=True)

print(f"\n总耗时 {time.time()-t0:.0f}s", flush=True)
