"""L7 分段求解: 先配出 (形状413, 色8), 再保持不变导航到锁。

单段 BFS 要摸到深度 60+ 才可能通关, 指数太贵。
分两段后每段深度减半:
  阶段1: BFS 到任意格上的 (413, 8) —— 实测 413 最早在深度 41 出现
  阶段2: 从该状态继续 BFS 到通关, 但**丢弃任何改变形状或颜色的转移**(踩机关就前功尽弃)
阶段2 若走不通(能量不够/到不了), 回阶段1 取下一个候选继续试。

两段都在真模拟器上实走, 解按构造即已验证。
"""
import copy, json, time
from collections import deque
import numpy as np
from wm import Percept, energy, load_env, panel_color, shape_bits
from arcengine import GameAction, ActionInput

ACTS = {a: getattr(GameAction, f"ACTION{a}") for a in (1, 2, 3, 4)}
START, T_SH, T_COL = (15, 19), 413, 8
MAX_CAND = 12                      # 阶段1 最多取几个候选去试阶段2


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
t0 = time.time()


def leg2(bg, e_start, prefix):
    """保持 (413,8) 不变, 导航到通关。返回完整动作串或 None。"""
    st = (p.key(np.array(bg.frame_stack()[-1])) if hasattr(bg, "frame_stack") else None)
    seen, dq, n = {}, deque([(bg, [], None, e_start)]), 0
    while dq:
        cur, path, cell, e = dq.popleft()
        for a in (1, 2, 3, 4):
            ch = clone(cur, clean); fr = raw(ch, a); n += 1
            if not fr.frame:
                continue
            if fr.levels_completed > base:
                return prefix + path + [a], n
            gg = np.array(fr.frame[-1])
            c2, sh2, col2, e2 = p.key(gg), shape_bits(gg), panel_color(gg), energy(gg)
            if (sh2, col2) != (T_SH, T_COL):
                continue                       # 踩了机关, 形状颜色被改, 这条废了
            if c2 == START and e2 > e and cell != START and cell is not None:
                continue                       # 死亡重生
            if e2 < 2 or seen.get(c2, -1) >= e2:
                continue
            seen[c2] = e2
            dq.append((ch, path + [a], c2, e2))
    return None, n


# ===== 阶段1: 搜到 (413, 8) =====
best = {s0: energy(g0)}
dq = deque([(game, [], s0, energy(g0))])
n = 0; cands = 0
print(f"阶段1: 从 {s0} 搜 ({T_SH},{T_COL})", flush=True)
while dq:
    bg, path, s, e = dq.popleft()
    for a in (1, 2, 3, 4):
        ch = clone(bg, clean); fr = raw(ch, a); n += 1
        if not fr.frame:
            continue
        if fr.levels_completed > base:
            sol = path + [a]
            print(f"*** 阶段1 途中直接通关! {len(sol)} 步", flush=True)
            json.dump({"level7_seq": sol}, open("l7_seq.json", "w")); raise SystemExit
        gg = np.array(fr.frame[-1])
        s2 = (p.key(gg), shape_bits(gg), panel_color(gg))
        e2 = energy(gg)
        if s2[0] == START and e2 > e and s[0] != START:
            continue
        if e2 < 2 or best.get(s2, -1) >= e2:
            continue
        best[s2] = e2

        if s2[1] == T_SH and s2[2] == T_COL:
            cands += 1
            print(f"候选{cands}: 深度{len(path)+1} 格{s2[0]} 能量{e2} "
                  f"{time.time()-t0:.0f}s -> 进阶段2", flush=True)
            sol, m = leg2(ch, e2, path + [a])
            if sol:
                print(f"*** L7 通关! 共 {len(sol)} 步 (阶段2 扩展{m}次)", flush=True)
                json.dump({"level7_seq": sol}, open("l7_seq.json", "w")); raise SystemExit
            print(f"  候选{cands} 阶段2 走不通(扩展{m}次), 继续", flush=True)
            if cands >= MAX_CAND:
                print("候选用尽, 放宽策略再来", flush=True); raise SystemExit
        dq.append((ch, path + [a], s2, e2))
    if n % 5000 == 0:
        print(f"  阶段1 扩展{n} 队列{len(dq)} 状态{len(best)} 深度~{len(path)} "
              f"{time.time()-t0:.0f}s", flush=True)
print("阶段1 队列空, 没搜到目标形状颜色", flush=True)
