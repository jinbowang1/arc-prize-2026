"""诊断: 用正常 step 重走 l6_seq.json, 定位与搜索(raw+deepcopy)的分岔点。"""
import json
import numpy as np
from wm import Percept, energy, load_env, panel_color, shape_bits, step

seq = json.load(open("l6_seq.json"))["level6_seq"]
game, f = load_env("solutions_l5.json")
base = f.levels_completed
p = Percept(np.array(f.frame[-1]))
print("L6 起始", f.levels_completed, flush=True)
for i, a in enumerate(seq, 1):
    f = step(game, a)
    if not f.frame:
        print(f"第{i}步死亡/无帧"); break
    g = np.array(f.frame[-1])
    s = (p.key(g), shape_bits(g), panel_color(g), energy(g))
    if i > len(seq) - 8 or i % 10 == 0:
        print(f"{i:>3} a={a} {s} lv={f.levels_completed}", flush=True)
    if f.levels_completed > base:
        print(f"*** 第{i}步通关"); break
else:
    print("走完未通关, 末状态", s)
