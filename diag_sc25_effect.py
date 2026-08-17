"""sc25: 表征是瞎的 —— 用"走两次"取代单步 peek, 能不能让它睁眼?

五倍预算的判据结果: best_first 扩展 2338 -> 12099(5.2 倍), 最深 9 -> 12 层,
**h 最好 3 -> 3 纹丝不动**。按判据这是**表征缺自由度**, 不是搜索不够强,
所以不该再加算力, 该去修表征。

表征坏在哪, 三个信号互相印证, 而且同源:
    [ReAct] 实体 0 个
    [model] 27 个动作: 可用 0 / 无效果 13
    [plan]  没有任何动作建成了常量覆盖表
这三层判"某动作改了哪些格"全都用**单步 peek**, 而 sc25 上单步 peek 看到的是
**上一个**动作的效果(perform_action 只注入, step() 才结算)。

修法不必偷调 step()(那会脱离官方语义)。已量到的事实:
    viaact(3,3) 逐格等于 truth(3)      <- 走两次 a, 得到的正是 a 的单次真效果
    viaact(3,1) 逐格等于 truth(1)
克隆体上试探免费, 所以"看 a 的效果"= fork -> act(a) -> act(a)。

⚠️注意这个画面对应的真实状态是"a 已结算一次 + 缓冲里还有一个 a", 不是"走了一次
a 的状态"。所以它只能用来**观测动作效果**(percept/model 建表征), 搜索仍走
act 语义(BFS 那边指纹加了 pending 之后已经正常)。

三问:
    ① 单步 peek 与"走两次"各自看到多少个有效动作
    ② 用"走两次"的 diff 做实体发现, 能不能分出实体
    ③ 动作效果是不是"常量覆盖表"(抽象层能不能用)
"""
from __future__ import annotations

import numpy as np

from harness.env import Action, Game, action_space
from harness.percept import analyze

GID = "sc25"
game, obs = Game.make(GID)
g0 = np.array(obs.grid)
sp = action_space(list(obs.actions))
scene = analyze(obs.grid)
clicks = [Action.click(c, r) for (r, c) in scene.targets]
acts = [Action.key(i) for i in sp["keys"]] + clicks
print(f"动作 {len(acts)} 个 = 按键 {len(sp['keys'])} + 点击 {len(clicks)}", flush=True)


def eff_single(a):
    """单步 peek(harness 现在的做法)"""
    o = game.peek(a)
    return None if o.dead else np.array(o.grid)


def eff_double(a):
    """走两次 a —— 官方语义下 a 的单次真效果"""
    n = game.fork()
    o1 = n.act(a)
    if o1.dead:
        return None
    o2 = n.act(a)
    return None if o2.dead else np.array(o2.grid)


print("\n[① 两种看法各自看到多少有效动作]", flush=True)
single_hits, double_hits = [], []
dmaps: dict[str, np.ndarray] = {}
for a in acts:
    gs, gd = eff_single(a), eff_double(a)
    ns = 0 if gs is None else int((gs != g0).sum())
    nd = 0 if gd is None else int((gd != g0).sum())
    if ns:
        single_hits.append(repr(a))
    if nd:
        double_hits.append(repr(a))
        dmaps[repr(a)] = (gd != g0)
    if nd != ns:
        print(f"    {repr(a):<14} 单步 {ns:>4} 格 -> 走两次 **{nd:>4}** 格", flush=True)
print(f"\n    单步 peek 看到有效动作: {len(single_hits)}/{len(acts)}", flush=True)
print(f"    走两次   看到有效动作: **{len(double_hits)}**/{len(acts)}", flush=True)

# ② 实体发现: 判据 = "总是一起变的格子才是同一实体"(签名 = 会改动它的动作集合)
print("\n[② 用走两次的 diff 做实体发现]", flush=True)
sig: dict[tuple, list[tuple[int, int]]] = {}
for (r, c) in np.ndindex(g0.shape):
    key = tuple(sorted(k for k, d in dmaps.items() if d[r, c]))
    if key:
        sig.setdefault(key, []).append((r, c))
print(f"    分出 {len(sig)} 个实体(按'会改动它的动作集合'归组):", flush=True)
for key, cells in sorted(sig.items(), key=lambda kv: -len(kv[1]))[:8]:
    rows = sorted({r for r, _ in cells}); cols = sorted({c for _, c in cells})
    print(f"      {len(cells):>4} 格 行{rows[0]}..{rows[-1]} 列{cols[0]}..{cols[-1]} "
          f"<- 被 {len(key)} 个动作改动 {list(key)[:4]}", flush=True)

# ③ 效果是不是常量(同一动作在不同状态下改同样的格 -> 抽象层可用)
print("\n[③ 效果是常量还是状态相关(抽象层能不能用)]", flush=True)
probe_acts = [a for a in acts if repr(a) in double_hits][:6]
for a in probe_acts:
    base = dmaps[repr(a)]
    same = 0
    trials = 0
    for lead in acts[:4]:
        n = game.fork()
        if n.act(lead).dead:
            continue
        n.act(lead)                       # 结算 lead
        before = np.array(n.act(a).grid)   # 注入 a
        o2 = n.act(a)
        if o2.dead:
            continue
        d = (np.array(o2.grid) != before)
        trials += 1
        same += int(np.array_equal(d, base))
    print(f"    {repr(a):<14} 在 {trials} 个不同前置状态下, 效果与开局相同的次数 {same}",
          flush=True)
