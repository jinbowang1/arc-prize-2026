"""诊断: 构型图 56 vs 1074 的矛盾。

第一档采集从当前构型 BFS **穷尽**得 56 个构型; anchor 轮从"当前构型走几步之后"
BFS 穷尽得 1074 个。anchor 是从当前构型走过去的 —— **可达集只会更小或相等**,
不可能从 56 涨到 1074。两种解释, 后果完全不同:

  A. 构型图真的不强连通, 且 anchor 打开了一片回不去的新空间
     -> 那 56 是真的, 规划必须在构型图上做(不能把笔当无序集合)
  B. 1074 是**虚高**的 —— 某个未掩的计数器让同一个构型被算成很多个
     -> 那是又一个计数器漏网, 跟 (63,55) 同类

判别很简单: 看 1074 里那些"新构型"两两之间差在哪几个格。差在一小片固定位置
= 计数器; 差在面板那种大片区域 = 真构型。
"""
from __future__ import annotations

import json
import time
from collections import deque

import numpy as np

from harness.canvas import _config_fp, _config_mask, _region, classify, collect_brushes
from harness.env import Action, Game, Obs, action_space
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
st = classify(game, obs, acts, BOX)
cmask = _config_mask(game, obs, st, BOX, rep.mask)


def bfs_cfgs(node: Game, ob: Obs, limit: int = 3000):
    """BFS 构型, 返回 {指纹: 一张代表帧}。"""
    fp0 = _config_fp(np.array(ob.grid), BOX, cmask)
    out = {fp0: np.array(ob.grid)}
    q = deque([(node.fork(), ob)])
    while q and len(out) < limit:
        nd, o = q.popleft()
        for a in st.adjusters:
            ch = nd.fork()
            o2 = ch.act(a)
            if o2.dead or o2.level != ob.level:
                continue
            fp = _config_fp(np.array(o2.grid), BOX, cmask)
            if fp in out:
                continue
            out[fp] = np.array(o2.grid)
            q.append((ch, o2))
    return out, (not q)


S0, done0 = bfs_cfgs(game, obs)
print(f"[A] 从当前构型 BFS: {len(S0)} 个构型{'(穷尽)' if done0 else '(截断)'} "
      f"| {time.time()-t0:.0f}s", flush=True)

# 拿第一档的库, 取一支笔的 seq 当 anchor(和 collect_brushes 里的定向一样)
brushes, complete, judged, total, ncfg = collect_brushes(
    game, obs, st, rep.mask, max_configs=2000, min_ratio=0.0)
print(f"[A] 第一档库 {len(brushes)} 支, 判 {judged}/{total}, 构型 {ncfg}", flush=True)
anchor = max(brushes, key=lambda b: len(b.seq)).seq
print(f"[B] anchor = {[str(a) for a in anchor]} ({len(anchor)} 步)", flush=True)

nd = game.fork()
cur = obs
ok = True
for a in anchor:
    cur = nd.act(a)
    if cur.dead or cur.level != obs.level:
        ok = False
        break
if not ok:
    raise SystemExit("anchor 走不通")

S1, done1 = bfs_cfgs(nd, cur)
print(f"[B] 从 anchor 终点 BFS: {len(S1)} 个构型{'(穷尽)' if done1 else '(截断)'} "
      f"| {time.time()-t0:.0f}s", flush=True)
print(f"[B] 交集 {len(set(S0) & set(S1))} | 只在 A {len(set(S0)-set(S1))} | "
      f"只在 B {len(set(S1)-set(S0))}", flush=True)

# 判别: 新构型之间差在哪几个格
extra = [S1[k] for k in list(set(S1) - set(S0))[:12]]
if len(extra) >= 2:
    diff = np.zeros((64, 64), dtype=bool)
    for i in range(len(extra)):
        for j in range(i + 1, len(extra)):
            diff |= (extra[i] != extra[j])
    diff &= cmask
    r0, r1, c0, c1 = BOX
    diff[r0:r1 + 1, c0:c1 + 1] = False
    cells = np.argwhere(diff)
    rows = sorted(set(int(r) for r, _ in cells))
    print(f"\n[判别] 新构型两两之间的差异格(答案区外, 未掩): {len(cells)} 个", flush=True)
    print(f"        行 {rows[:12]}", flush=True)
    print(f"        前 16 格 {cells[:16].tolist()}", flush=True)
    if len(cells) <= 12 and rows and min(rows) >= 60:
        print("        -> 集中在底部一小片 = **又一个计数器漏网**(同 (63,55) 那类)",
              flush=True)
    else:
        print("        -> 分布在大片区域 = 真构型差异, 构型图确实不强连通", flush=True)
print(f"\n总耗时 {time.time()-t0:.0f}s", flush=True)
