"""L7: 直接在真模拟器上做 BFS —— 边建模边找通关。

L7 的离线模型太稀疏(282 条转移, 只发现 2 个机关格), 不足以规划。
但 clone+raw 试探不消耗真实进度, 所以可以拿真机当模型用:
状态 (格, 形状, 颜色), 按最大能量支配剪枝, 广度优先 => 找到的第一个解即最短且已验证。

锁: 色8 + 形状413 (validate_lock 在 (49,28) 读到)。
"""
import copy, json, time
from collections import deque
import numpy as np
from wm import Percept, energy, load_env, panel_color, shape_bits
from arcengine import GameAction, ActionInput

ACTS = {a: getattr(GameAction, f"ACTION{a}") for a in (1, 2, 3, 4)}
START = (15, 19)


def raw(g, a): return g.perform_action(ActionInput(id=ACTS[a], data={}), raw=True)


def clone(g, clean):
    g._clean_levels = None; g2 = copy.deepcopy(g); g._clean_levels = clean
    g2._clean_levels = clean; return g2


game, f = load_env("solutions_l6.json")
base = f.levels_completed
clean = game._clean_levels
p = Percept(np.array(f.frame[-1]))
g0 = np.array(f.frame[-1])
s0 = (p.key(g0), shape_bits(g0), panel_color(g0))
e0 = energy(g0)
print(f"L7 起点 {s0} 能量 {e0}", flush=True)

best = {s0: e0}                      # (格,形状,颜色) -> 到达过的最高能量
dq = deque([(game, [], s0, e0)])
n = 0; t0 = time.time(); cells = {s0[0]}; shapes = {s0[1]}; cols = {s0[2]}
while dq:
    bg, path, s, e = dq.popleft()
    for a in (1, 2, 3, 4):
        ch = clone(bg, clean); fr = raw(ch, a); n += 1
        if not fr.frame:
            continue
        if fr.levels_completed > base:
            sol = path + [a]
            print(f"*** L7 通关! {len(sol)} 步", flush=True)
            json.dump({"level7_seq": sol}, open("l7_seq.json", "w"))
            raise SystemExit
        gg = np.array(fr.frame[-1])
        s2 = (p.key(gg), shape_bits(gg), panel_color(gg))
        e2 = energy(gg)
        if s2[0] == START and e2 > e and s[0] != START:
            continue                              # 死亡重生, 剪掉
        if e2 < 2 or best.get(s2, -1) >= e2:
            continue
        best[s2] = e2
        cells.add(s2[0]); shapes.add(s2[1]); cols.add(s2[2])
        dq.append((ch, path + [a], s2, e2))
    if n % 2000 == 0:
        print(f"扩展{n} 队列{len(dq)} 状态{len(best)} 深度~{len(path)} "
              f"格{len(cells)} 形状{len(shapes)} 色{sorted(cols)} {time.time()-t0:.0f}s", flush=True)
print(f"队列空: 状态{len(best)} 格{len(cells)} 形状{sorted(shapes)} 色{sorted(cols)}", flush=True)
