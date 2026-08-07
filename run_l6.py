"""L6 执行: 规则式规划 -> 真机逐步验证 -> 进上锁。"""
import json
import numpy as np
from plan_l6 import load, plan
from wm import Percept, energy, load_env, panel_color, shape_bits, step

move, fam, pickups = load()
print("格级转移", len(move), "换族条目", len(fam), "补给点", sorted(pickups))
game, f = load_env("solutions_l5.json")
base = f.levels_completed
p = Percept(np.array(f.frame[-1]))
g = np.array(f.frame[-1])
start = (p.key(g), shape_bits(g), panel_color(g))
print("起始", start, "能量", energy(g))

r = plan(move, fam, pickups, start, energy(g), (30, 54), 413, 8)
if r is None:
    print("无解: 规则式规划也找不到")
    raise SystemExit
path, e = r
print(f"规划 {len(path)} 步, 余能量 {e}")
seq = []
for a in path:
    f = step(game, a); seq.append(a)
    if not f.frame:
        print("中途死亡"); raise SystemExit
g = np.array(f.frame[-1])
print("到位:", p.key(g), shape_bits(g), panel_color(g), energy(g))
f = step(game, 2); seq.append(2)
if f.frame and f.levels_completed > base:
    print(f"*** L6 通关! 本关 {len(seq)} 步")
    json.dump({"level6_seq": seq}, open("l6_seq.json", "w"))
else:
    g = np.array(f.frame[-1]) if f.frame else None
    print("上锁未开:", (p.key(g), shape_bits(g), panel_color(g)) if g is not None else "死亡")
