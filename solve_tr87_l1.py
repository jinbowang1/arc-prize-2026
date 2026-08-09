"""tr87 L1: clone 树 DFS 穷举 5 位 x 7 符号, 找到即在真机重放最短序列。

机制(已实测): ACTION1/2 = 当前位符号正/反循环(7 种), ACTION3/4 = 光标左/右移(5 位循环),
无提交键 -> 5 位全对即自动过关。clone 试探不耗真实步数。
"""
import copy, sys, time
import numpy as np
import arc_agi
from arcengine import GameAction, ActionInput

ACTS = {i: getattr(GameAction, f"ACTION{i}") for i in range(1, 8)}

def raw(g, a):
    return g.perform_action(ActionInput(id=ACTS[a], data={}), raw=True)

def clone(g):
    clean = getattr(g, "_clean_levels", None)
    g._clean_levels = None; g2 = copy.deepcopy(g)
    g._clean_levels = clean; g2._clean_levels = clean
    return g2

arc = arc_agi.Arcade()
env = arc.make("tr87")
f = env.reset()
game = env._game
print(f"开局 levels={f.levels_completed}/{f.win_levels}")

stats = {"clones": 0, "acts": 0}
t0 = time.time()

def dfs(g, depth, path):
    """g: 光标已在位 depth(0-based), 该位处于初始符号(k=0)。path=已定各位的 k。"""
    for k in range(7):
        if k > 0:
            fr = raw(g, 1); stats["acts"] += 1
            if fr.levels_completed > 0:
                return path + [k]
        if depth < 4:
            ch = clone(g); stats["clones"] += 1
            fr = raw(ch, 4); stats["acts"] += 1
            if fr.levels_completed > 0:
                return path + [k]
            r = dfs(ch, depth + 1, path + [k])
            if r:
                return r
    return None

sol = dfs(clone(game), 0, [])
dt = time.time() - t0
print(f"穷举结果: {sol}  (clones={stats['clones']} acts={stats['acts']} {dt:.1f}s)")
if not sol:
    sys.exit("未找到解 — 过关判定假设有误, 需要重新勘探")

# 真机最短序列: 每位 k 用 min(正循环 k 次, 反循环 7-k 次); 位间 ACTION4 右移(仅在后面还有要改的位时)
last_nonzero = max((i for i, k in enumerate(sol) if k > 0), default=-1)
seq = []
for i, k in enumerate(sol):
    if k > 0:
        seq += [1] * k if k <= 3 else [2] * (7 - k)
    if i < last_nonzero:
        seq.append(4)
print(f"真机序列({len(seq)}步 vs 人类54): {seq}")

for a in seq:
    f = env.step(ACTS[a])
print(f"执行后: state={f.state.name} levels={f.levels_completed} 动作计数={env._game.action_counter if hasattr(env._game,'action_counter') else '?'}")
if f.levels_completed >= 1:
    import json
    json.dump({"game": "tr87", "l1_seq": seq, "l1_combo": sol}, open("tr87_l1.json", "w"))
    print("L1 解已存 tr87_l1.json")
    g2 = np.array(f.frame[-1])
    CH = ".123456789ABCDEF"
    print("\n=== L2 开局帧 ===")
    for r in range(64):
        print(f"{r:>3}  " + "".join(CH[v] for v in g2[r]))
