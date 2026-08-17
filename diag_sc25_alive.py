"""sc25 第二问: 开局所有动作都改 0 格 —— 这游戏到底醒着没有?

第一轮勘探否掉了 harness 自己给的归因("掩码掩掉了游戏区域"): 掩码掩了 **0 格**,
而 27 个动作**各自都改 0 格**, 带不带掩码都只有 1 个后继指纹。
但裸跑报"动作预算 52(GAME_OVER)" —— 动作是被接受并计数的, 只是画面不动。

四个候选解释, 挨个测:
    ① 要连续动作才有反应(单步 peek 看不见)
    ② 有动画帧: 同一动作连发, 画面隔几帧才更新
    ③ 开局是标题/等待画面, 要某个特定动作才进入游戏
    ④ available_actions 随状态变, 开局这批根本不是有效动作

⚠️全程在克隆体上试, 真机不动 —— 52 步的硬预算浪费不起。
"""
from __future__ import annotations

import numpy as np

from harness.env import Action, Game, action_space

GID = "sc25"
game, obs = Game.make(GID)
g0 = np.array(obs.grid)
sp = action_space(list(obs.actions))
print(f"开局: level={obs.level} state={getattr(obs, 'state', '?')} "
      f"available={sorted(obs.actions)}", flush=True)

# ── ①② 同一动作连发, 看第几步开始动 ──
print("\n[①② 同一动作连发 12 次, 画面第几步开始变]", flush=True)
for key in sp["keys"]:
    node = game.fork()
    prev = g0
    marks = []
    for step in range(12):
        o = node.act(Action.key(key))
        g = np.array(o.grid)
        marks.append(str(int((g != prev).sum())))
        prev = g
        if o.dead:
            marks.append("DEAD")
            break
    total = int((prev != g0).sum())
    print(f"    A{key}: 逐步 diff = {' '.join(marks)} | 12 步后累计改 {total} 格", flush=True)

# ── ③ 全动作单发一遍, 有没有哪个能叫醒它(含 A5/A7 这类没在 available 里的) ──
print("\n[③ 试所有动作 id(含未在 available 列出的)]", flush=True)
for aid in range(1, 8):
    node = game.fork()
    try:
        a = Action.key(aid) if aid <= 5 else Action.click(32, 32)
        o = node.act(a)
        n = int((np.array(o.grid) != g0).sum())
        print(f"    id={aid} {repr(a):<12} 改 {n:>4} 格 | level={o.level} "
              f"dead={o.dead} available={sorted(o.actions)}", flush=True)
    except Exception as ex:
        print(f"    id={aid} 抛错 {type(ex).__name__}: {str(ex)[:60]}", flush=True)

# ── ④ 点击扫全屏(粗网格), 有没有任何一格点了有反应 ──
print("\n[④ 点击粗扫全屏 8x8 网格]", flush=True)
hits = []
for r in range(2, 64, 8):
    for c in range(2, 64, 8):
        node = game.fork()
        o = node.act(Action.click(c, r))
        n = int((np.array(o.grid) != g0).sum())
        if n:
            hits.append((r, c, n))
print(f"    有反应的点: {len(hits)} 个 {hits[:10]}", flush=True)

# ── 兜底: 连发混合动作 30 步, 看会不会自己动(是不是在等时间) ──
print("\n[⑤ 混合连发 30 步, 画面会不会动]", flush=True)
node = game.fork()
prev = g0
changed_at = []
for step in range(30):
    a = Action.key(sp["keys"][step % len(sp["keys"])])
    o = node.act(a)
    g = np.array(o.grid)
    if int((g != prev).sum()):
        changed_at.append((step, int((g != prev).sum())))
    prev = g
    if o.dead:
        changed_at.append((step, "DEAD"))
        break
print(f"    发生变化的步: {changed_at[:12]}", flush=True)
print(f"    30 步后 level={o.level} dead={o.dead} 累计改 {int((prev != g0).sum())} 格", flush=True)
