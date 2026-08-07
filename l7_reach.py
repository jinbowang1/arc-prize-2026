"""L7 可行性探针: 只问"配出 形状413+色8 最少要几步", 不管站在哪。

结果决定主搜索该用 BFS 还是必须换引导式搜索: 若配色配形状本身就要几十步,
那"配好 + 走到锁"的总深度会让无引导 BFS 吃不消。
"""
import copy, time
from collections import deque
import numpy as np
from wm import Percept, energy, load_env, panel_color, shape_bits
from arcengine import GameAction, ActionInput

ACTS = {a: getattr(GameAction, f"ACTION{a}") for a in (1, 2, 3, 4)}
START, T_SH, T_COL = (15, 19), 413, 8


def raw(g, a): return g.perform_action(ActionInput(id=ACTS[a], data={}), raw=True)


def clone(g, clean):
    g._clean_levels = None; g2 = copy.deepcopy(g); g._clean_levels = clean
    g2._clean_levels = clean; return g2


game, f = load_env("solutions_l6.json")
clean = game._clean_levels
p = Percept(np.array(f.frame[-1]))
g0 = np.array(f.frame[-1])
s0 = (p.key(g0), shape_bits(g0), panel_color(g0))
best = {s0: energy(g0)}
dq = deque([(game, [], s0, energy(g0))])
n = 0; t0 = time.time(); seen_sh = {s0[1]}
while dq:
    bg, path, s, e = dq.popleft()
    for a in (1, 2, 3, 4):
        ch = clone(bg, clean); fr = raw(ch, a); n += 1
        if not fr.frame:
            continue
        gg = np.array(fr.frame[-1])
        s2 = (p.key(gg), shape_bits(gg), panel_color(gg))
        e2 = energy(gg)
        if s2[1] == T_SH and s2[2] == T_COL:
            print(f"*** 配出 ({T_SH},{T_COL}) 需 {len(path)+1} 步, 在格 {s2[0]}, 余能量 {e2}", flush=True)
            print(f"    路径 {path + [a]}", flush=True)
            raise SystemExit
        if s2[0] == START and e2 > e and s[0] != START:
            continue
        if e2 < 2 or best.get(s2, -1) >= e2:
            continue
        best[s2] = e2; seen_sh.add(s2[1])
        dq.append((ch, path + [a], s2, e2))
    if n % 4000 == 0:
        print(f"扩展{n} 队列{len(dq)} 状态{len(best)} 深度~{len(path)} "
              f"见过形状{len(seen_sh)} 含413={T_SH in seen_sh} {time.time()-t0:.0f}s", flush=True)
print(f"队列空: 配不出 ({T_SH},{T_COL})。见过形状 {sorted(seen_sh)}", flush=True)
