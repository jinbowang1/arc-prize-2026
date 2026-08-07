"""L7 机制勘探: 形状可达集到底有多大、哪些格是机关、413 够不够得着。

盲搜到深度 48 只长出 24 种形状且不含目标 413, 说明问题不在搜索深度而在机制没摸清。
这里不求解, 只求三件事:
  1. 形状可达集的完整列表(跑到队列空为止)
  2. 每个机关格的 (旧形状,旧色) -> (新形状,新色) 映射
  3. 413 到底能不能长出来; 长不出来就说明还有没触发的机关
"""
import copy, json, time
from collections import deque
import numpy as np
from wm import Percept, energy, load_env, panel_color, shape_bits
from arcengine import GameAction, ActionInput

ACTS = {a: getattr(GameAction, f"ACTION{a}") for a in (1, 2, 3, 4)}
START, T_SH, T_COL = (15, 19), 413, 8


def show(b):
    return "/".join("".join("X" if b >> (i * 3 + j) & 1 else "." for j in range(3))
                    for i in range(3))


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
dq = deque([(game, s0, energy(g0), 0)])
mech = {}                      # 格 -> {(旧形状,旧色) -> (新形状,新色)}
shapes, n, t0 = {s0[1]}, 0, time.time()
while dq:
    bg, s, e, dep = dq.popleft()
    for a in (1, 2, 3, 4):
        ch = clone(bg, clean); fr = raw(ch, a); n += 1
        if not fr.frame:
            continue
        gg = np.array(fr.frame[-1])
        s2 = (p.key(gg), shape_bits(gg), panel_color(gg))
        e2 = energy(gg)
        if s2[0] == START and e2 > e and s[0] != START:
            continue                                   # 死亡重生
        if (s2[1], s2[2]) != (s[1], s[2]):             # 形状或颜色变了 = 踩到机关
            mech.setdefault(s2[0], {})[(s[1], s[2])] = (s2[1], s2[2])
            if s2[1] not in shapes:
                shapes.add(s2[1])
                print(f"  新形状 {s2[1]}={show(s2[1])} 于机关格 {s2[0]} "
                      f"(由 {s[1]}={show(s[1])} 变来) 深度{dep+1}", flush=True)
        if e2 < 2 or best.get(s2, -1) >= e2:
            continue
        best[s2] = e2
        dq.append((ch, s2, e2, dep + 1))
    if n % 5000 == 0:
        print(f"扩展{n} 队列{len(dq)} 状态{len(best)} 深度~{dep} 形状{len(shapes)} "
              f"机关格{len(mech)} 含413={T_SH in shapes} {time.time()-t0:.0f}s", flush=True)

print(f"\n=== 穷尽 ===\n形状可达集 {len(shapes)} 种:")
for s in sorted(shapes):
    print(f"  {s}={show(s)}" + ("   <== 目标" if s == T_SH else ""))
print(f"\n413 可达: {T_SH in shapes}")
print(f"\n机关格 {len(mech)} 个:")
for cell, m in sorted(mech.items()):
    print(f"  {cell}:")
    for k, v in sorted(m.items()):
        print(f"     {k} -> {v}")
json.dump({"shapes": sorted(shapes),
           "mech": {str(k): {str(a): list(b) for a, b in v.items()} for k, v in mech.items()}},
          open("l7_mech.json", "w"))
