"""tr87 通用 DFS 穷举(任意位数): clone 树逐位枚举环, 过关即停。"""
import copy, json, sys, time
import numpy as np
import arc_agi
from arcengine import GameAction, ActionInput
from parse_tr87 import parse
from solve_tr87 import clone, raw, tup, ACTS

arc = arc_agi.Arcade()
env = arc.make("tr87")
f = env.reset()
sols = json.load(open("tr87_solutions.json"))
for seq in sols["seqs"]:
    for a in seq:
        f = env.step(ACTS[a])
game = env._game
level = f.levels_completed + 1
g = np.array(f.frame[-1])
_, _, ans0, _ = parse(g)
N = len(ans0)
print(f"L{level}: {N} 位, DFS 穷举开始", flush=True)

stats = {"clones": 0, "acts": 0, "t0": time.time()}

def dfs(g, depth, path):
    for k in range(7):
        if k > 0:
            fr = raw(g, 1); stats["acts"] += 1
            if fr.levels_completed >= level:
                return path + [k]
        if depth < N - 1:
            ch = clone(g); stats["clones"] += 1
            if stats["clones"] % 5000 == 0:
                print(f"  clones={stats['clones']} acts={stats['acts']} "
                      f"{time.time()-stats['t0']:.0f}s path={path+[k]}", flush=True)
            fr = raw(ch, 4); stats["acts"] += 1
            if fr.levels_completed >= level:
                return path + [k]
            r = dfs(ch, depth + 1, path + [k])
            if r:
                return r
    return None

sol = dfs(clone(game), 0, [])
print(f"结果: {sol} (clones={stats['clones']} acts={stats['acts']} {time.time()-stats['t0']:.0f}s)", flush=True)
if not sol:
    sys.exit(1)

# 真机最短序列
rings_len = 7
seq = []
last = max((i for i, k in enumerate(sol) if k > 0), default=-1)
for i, k in enumerate(sol):
    if k > 0:
        seq += [1] * k if k <= 3 else [2] * (7 - k)
    if i < last:
        seq.append(4)
for a in seq:
    f = env.step(ACTS[a])
print(f"真机: levels={f.levels_completed} ({len(seq)}步)", flush=True)
if f.levels_completed >= level:
    sols["seqs"].append(seq)
    json.dump(sols, open("tr87_solutions.json", "w"))
    print("已存")
