"""这一局是不是"画布类"游戏? —— 决定能不能用语义掩码。

改进 #1 失败的根因之一: 我无差别地给所有局套语义掩码, 而 **ls20 是导航游戏**
(钥匙移动去匹配锁), 根本没有"画布", 掩完只剩 200 格 -> 26 个状态就假穷尽。

判据(可测): **画布类 = 提交动作"只改答案区, 不改别处"**。
    cd82 画笔只涂画布 ✓        sc25 点击只改那一格 ✓
    ls20 按键让钥匙**移动** ✗  (同时改原位置和新位置, 溢出答案区)

对每局: 取 propose_prompt_answer 的提议, 按"提交集同类且最多"选一个,
然后看**提交动作改动的格子有多少落在答案区之外**(已掩的计数器不算)。
溢出比例低 => 画布类, 可以用语义掩码; 高 => 不是, 别用。
"""
from __future__ import annotations

import sys

import numpy as np

from harness.canvas import classify
from harness.env import Action, Game, action_space
from harness.hypo import propose_prompt_answer
from harness.model import collect_states
from harness.percept import analyze, mutable_over_states
from harness.probe import run_probe

GAMES = sys.argv[1:] or ["cd82", "ft09", "sc25", "r11l", "ls20", "tr87"]
print(f"{'game':6} {'答案区':<22} {'提交':>4} {'区外溢出':>8}  判定")
for gid in GAMES:
    try:
        game, obs = Game.make(gid)
        sp = action_space(list(obs.actions))
        sc = analyze(obs.grid)
        clicks = [Action.click(c, r) for (r, c) in sc.targets]
        acts = [Action.key(i) for i in sp["keys"]] + clicks
        game.detect_lag(acts)
        rep = run_probe(game, obs, sp["kind"], sp["keys"], clicks)
        states = collect_states(game, obs, acts, 5)
        mut = mutable_over_states(
            [lambda a, c=c: np.array(c.effect(a).grid) for c, _ in states],
            [np.array(o.grid) for _, o in states], acts) & rep.mask
        props = propose_prompt_answer(np.array(obs.grid), mut, sc.bg)
        pick = st = None
        best = 0
        for h in props:
            t = classify(game, obs, acts, h.a)
            subs = [repr(a) for a in t.submitters]
            nc = sum(1 for r in subs if r.startswith("A6"))
            if subs and nc in (0, len(subs)) and len(subs) > best:
                pick, st, best = h, t, len(subs)
        if pick is None:
            print(f"{gid:6} {'(认不出)':<22} {'-':>4} {'-':>8}  非画布类")
            continue
        g0 = np.array(obs.grid)
        r0, r1, c0, c1 = pick.a
        inside = outside = 0
        for a in st.submitters:
            o = game.effect(a)
            if o.dead:
                continue
            d = (np.array(o.grid) != g0)
            if rep.mask is not None:
                d &= rep.mask                     # 计数器不算
            box = np.zeros_like(d); box[r0:r1+1, c0:c1+1] = True
            inside += int((d & box).sum())
            outside += int((d & ~box).sum())
        # 🚨inside == 0 必须判非画布类: 提交动作**根本没改答案区**,
        # 而 outside/max(1,0) 会算出 0% 溢出 —— **"没有证据"被当成"证据完美"**。
        # ls20 实测就栽在这: 3 个"提交"改 0 格, 却被判成最干净的画布类。
        # 🚨真正的判据是"**构型 x 提交**结构": 提交集与调整集**都要非空且都有效**。
        # ls20 栽在这: 钥匙移动会改答案区, classify 把移动键全判成"提交",
        # adjusters 空 -> 构型区空 -> 语义掩码只剩画布, **钥匙位置被整个掩掉**
        # -> 26 个状态就假穷尽(改进 #1 让 ls20 从 1/7 掉到 0/7 的直接原因)。
        n_adj = 0
        for a in st.adjusters:
            o = game.effect(a)
            if not o.dead and not np.array_equal(np.array(o.grid), g0):
                n_adj += 1
        if inside == 0:
            ratio, verdict = 1.0, "非画布类(提交没改答案区, **无证据**)"
        elif n_adj == 0:
            ratio = outside / (inside + outside)
            verdict = "**非**画布类(调整集为空 -> 没有构型这一维)"
        else:
            ratio = outside / (inside + outside)
            verdict = "**画布类**" if ratio < 0.2 else "非画布类(溢出大)"
        print(f"{gid:6} {str(pick.a):<22} {len(st.submitters):>4} "
              f"{ratio:>7.0%}  {verdict} (改内 {inside} 外 {outside}, 有效调整 {n_adj})")
    except Exception as e:
        print(f"{gid:6} 出错 {type(e).__name__}: {str(e)[:40]}")
