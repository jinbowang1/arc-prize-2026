"""诊断: canvas 的骨架到底成不成立?

骨架是这一整套方法的地基, 一共两句话:
    ① 提交只改画布, 不改构型
    ② 调整只改构型, 与画布无关
② 从来没被单独验过。而图上规划已经保证"从当前构型可达", 执行时真机仍报
"走不到" —— 那就只剩一个解释: **构型转移依赖画布**, 即 ② 不成立。

判据干净且直接:
    同构型、不同画布, 走同一条调整序列, 终点构型是否相同。

两个底靠"提交一次"造(提交不改构型, ① 已在开局实测 60/60)。
在多个起点、多条序列上问 —— 只在一个状态上问是这个项目栽过六次的老毛病。

如果 ② 不成立, 那么"构型"这个抽象在 cd82 L3 上就是错的, canvas 这条路
应该停下来换方法, 而不是继续加规划器。
"""
from __future__ import annotations

import json
import time

import numpy as np

from harness.canvas import _config_fp, _config_mask, _region, classify
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
print(f"答案区 {BOX} | 提交 {len(st.submitters)} 调整 {len(st.adjusters)} "
      f"| 构型掩码有效 {int(cmask.sum())} 格", flush=True)


def walk(g: Game, o: Obs, seq):
    nd = g.fork()
    cur = o
    for a in seq:
        cur = nd.act(a)
        if cur.dead or cur.level != o.level:
            return None, None
    return nd, cur


def fp(o: Obs) -> bytes:
    return _config_fp(np.array(o.grid), BOX, cmask)


# 起点: 当前构型, 以及走几步之后的几个构型
starts: list[tuple[Game, Obs, str]] = [(game.fork(), obs, "开局")]
for k, a in enumerate(st.adjusters[:3]):
    nd, o = walk(game, obs, [a])
    if nd is not None:
        starts.append((nd, o, f"走过 {a}"))

SEQS = [
    st.adjusters[:1], st.adjusters[:2], st.adjusters[:3],
    st.adjusters[1:4], st.adjusters[2:6],
]

print("\n[② 调整动作的效果与画布无关?]", flush=True)
ok = bad = skip = 0
# 健全性: "终点构型相同"必须是有信息的判断。
#   moved = 这条调整序列真把构型挪动了(否则比的是同一个点, 恒相等)
#   seen  = 全程见过多少个互不相同的构型指纹(退化成 1 就说明指纹是常量)
moved = still = 0
seen: set[bytes] = set()
for nd, o, name in starts:
    # 同构型的第二个画布: 提交一次(提交不改构型)
    alt = None
    for sub in st.submitters:
        n2, o2 = walk(nd, o, [sub])
        if n2 is None:
            continue
        if fp(o2) != fp(o):
            continue                      # 这个提交动作改了构型, 换一个
        if np.array_equal(_region(np.array(o2.grid), BOX),
                          _region(np.array(o.grid), BOX)):
            continue                      # 画布没变(幂等), 造不出第二个底
        alt = (n2, o2)
        break
    if alt is None:
        print(f"  {name}: 造不出同构型不同画布的一对, 跳过", flush=True)
        continue
    n2, o2 = alt
    d_canvas = int((_region(np.array(o.grid), BOX)
                    != _region(np.array(o2.grid), BOX)).sum())
    seen.add(fp(o))
    print(f"  {name}: 两底画布差 {d_canvas} 格", flush=True)
    for seq in SEQS:
        if not seq:
            continue
        a1, c1 = walk(nd, o, seq)
        a2, c2 = walk(n2, o2, seq)
        if a1 is None or a2 is None:
            skip += 1
            continue
        same = fp(c1) == fp(c2)
        ok += same
        bad += (not same)
        seen.add(fp(c1))
        seen.add(fp(c2))
        if fp(c1) == fp(o):
            still += 1          # 这条序列压根没挪动构型, 这组比对无信息
        else:
            moved += 1
        if not same:
            g1, g2 = np.array(c1.grid), np.array(c2.grid)
            d = (g1 != g2)
            r0, r1, c0, c1_ = BOX
            d[r0:r1 + 1, c0:c1_ + 1] = False
            cells = np.argwhere(d & cmask)
            rows = sorted(set(int(r) for r, _ in cells))
            print(f"  ❌ {name} + {len(seq)} 步调整: 终点构型**不同** "
                  f"(两底画布差 {d_canvas} 格) -> 构型差 {len(cells)} 格, 行 {rows[:8]}",
                  flush=True)

print(f"\n结论: 终点构型相同 {ok} / 不同 {bad} / 走不通跳过 {skip}", flush=True)
print(f"区分力: 互不相同的构型指纹 {len(seen)} 个 | "
      f"真挪动了构型的比对 {moved} 组, 原地未动 {still} 组", flush=True)
if len(seen) <= 1 or moved == 0:
    print("  🚨 **本次诊断无效, 不要采信上面那行结论**: 指纹没有区分力"
          "(或所有调整序列都没挪动构型) —— 一切都'相同'是因为一切都是同一个点。"
          "\n     先修 _config_mask / 换调整序列, 再重跑。", flush=True)
elif bad == 0 and ok > 0:
    print("  ✅ ② 成立 —— 调整动作的效果与画布无关, 骨架没问题,"
          "\n     那么'走不到'另有原因(adj 记录不全 / 找路的动作表不够)", flush=True)
elif bad:
    print("  ❌ **② 不成立 —— 构型转移依赖画布**。"
          "\n     '构型'这个抽象在这一关上是错的: 同一条调整序列在不同画布下到不了同一个地方,"
          "\n     所以采集时记下的构型图, 执行时(画布已变)根本不适用。"
          "\n     -> canvas 这条路要么换表征, 要么停。", flush=True)
print(f"总耗时 {time.time()-t0:.0f}s", flush=True)
