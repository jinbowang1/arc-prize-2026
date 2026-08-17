"""sc25 机制勘探: 裸跑 1 秒就报"深度<=1 队列穷尽", 先查指纹/掩码, 别加搜索。

裸跑给的信号:
    实体 0 个 | 动作预算 52(GAME_OVER) | 27 个动作: 可用 0 / 状态相关 14 / 无效果 13
    BFS 扩展 1 节点、见过 1 状态、最深 0 层 -> harness 自己报"几乎必然是指纹坏了"

嫌疑是 cd82 上记过的同款病: **纯交集判 HUD 会误伤** —— 一关里所有动作恰好都改
同一片游戏区, 那片就被当计数器整片掩掉, 于是所有子状态指纹相同、被全部去重。
正解是"**变成什么与按了什么无关**才是计数器", 不是"每个动作都改它"。

所以这里只回答四个问题, 一个都不猜:
    ① probe 掩掉了哪些格, 掩掉的是不是真计数器
    ② 每个动作各自改了哪些格(帧 diff), 是不是全都落在被掩区
    ③ 去掉掩码之后, 子状态指纹还相不相同
    ④ 屏幕上有哪些连通块 / 哪块是"我能改的" / 哪块是"改不动但有内容的"
"""
from __future__ import annotations

import numpy as np

from harness.env import Action, Game, action_space
from harness.hypo import propose_prompt_answer
from harness.model import collect_states
from harness.percept import analyze, mutable_over_states
from harness.probe import run_probe

GID = "sc25"
game, obs = Game.make(GID)
g0 = np.array(obs.grid)
sp = action_space(list(obs.actions))
scene = analyze(obs.grid)
clicks = [Action.click(c, r) for (r, c) in scene.targets]
acts = [Action.key(i) for i in sp["keys"]] + clicks
print(f"动作空间 {sp['kind']} | 按键 {sp['keys']} | 点击目标 {len(clicks)} 个", flush=True)
print(f"画面 {g0.shape} 用到的颜色 {sorted(set(g0.flatten().tolist()))}", flush=True)

rep = run_probe(game, obs, sp["kind"], sp["keys"], clicks)
masked = ~rep.mask
print(f"\n[① probe 掩码] 掩掉 {int(masked.sum())} 格 / {g0.size}", flush=True)
if masked.any():
    rows = sorted(set(int(r) for r, _ in np.argwhere(masked)))
    cols = sorted(set(int(c) for _, c in np.argwhere(masked)))
    print(f"    行范围 {rows[0]}..{rows[-1]}({len(rows)} 行) 列范围 {cols[0]}..{cols[-1]}({len(cols)} 列)",
          flush=True)

# ② 每个动作单独的帧 diff
print("\n[② 各动作改了哪些格]", flush=True)
diffs: dict[str, np.ndarray] = {}
for a in acts:
    o = game.peek(a)
    if o.dead:
        print(f"    {repr(a):<14} 致死", flush=True)
        continue
    d = (np.array(o.grid) != g0)
    diffs[repr(a)] = d
    n = int(d.sum())
    inside = int((d & masked).sum())
    tag = ""
    if n and inside == n:
        tag = "  ⚠️改动**全部**落在被掩区 -> 这个动作在指纹上等于 noop"
    print(f"    {repr(a):<14} 改 {n:>4} 格, 其中 {inside:>4} 格在被掩区{tag}", flush=True)

# ③ 掩码是不是把"游戏区"掩掉了: 不用掩码时, 各动作的后继指纹还一样吗
print("\n[③ 指纹去重是不是被掩码毁掉的]", flush=True)
for use_mask in (True, False):
    fps = set()
    for a in acts:
        o = game.peek(a)
        if o.dead:
            continue
        g = np.array(o.grid)
        fps.add((g * rep.mask).tobytes() if use_mask else g.tobytes())
    print(f"    {'带掩码' if use_mask else '不带掩码'}: {len(acts)} 个动作产生 "
          f"**{len(fps)}** 个互不相同的后继指纹", flush=True)

# ④ 场景结构 + 因果目标
print("\n[④ 场景与目标]", flush=True)
print(f"    连通块 {len(scene.targets)} 个, 背景色 {scene.bg}", flush=True)
states = collect_states(game, obs, acts, 5)
mut = mutable_over_states([lambda a, c=c: np.array(c.peek(a).grid) for c, _ in states],
                          [np.array(o.grid) for _, o in states], acts) & rep.mask
print(f"    可变格(多状态并集) {int(mut.sum())} 格", flush=True)
for h in propose_prompt_answer(g0, mut, scene.bg)[:5]:
    print(f"    目标假设: {h.text() if hasattr(h, 'text') else h}", flush=True)
