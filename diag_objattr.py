"""ls20: h=3 是"像素差 3 格但语义不对"吗? 对象属性能不能拆穿它?

目标 region_match((35,44,29,38) 要等于 (5,14,19,28)) 只比像素。
ls20 的真语义是**钥匙的形状 + 颜色**(当年手工破关用的就是这个)。
先看这两块区域的实际内容, 判断"像素接近"是否等于"语义接近"。
"""
import numpy as np
from harness.env import Action, Game, action_space
from harness.percept import analyze

PAL = " .:-=+*#%@$&XO<>"
game, obs = Game.make("ls20")
sp = action_space(list(obs.actions))
sc = analyze(obs.grid)
acts = [Action.key(i) for i in sp["keys"]]
game.detect_lag(acts)

# 走 L1 的在案解进 L2
import json
seq = json.load(open("solutions.json"))["seq"]
per = [22]          # L1 人类基准; 在案解逐关步数见 solutions.json
o = obs
n = 0
for t in seq:
    a = Action.key(int(t)) if isinstance(t, int) else None
    if a is None:
        break
    o2 = game.act(a)
    n += 1
    if o2.level > o.level:
        o = o2
        break
    o = o2
print(f"走 {n} 步进到 level={o.level}")
g = np.array(o.grid)

A = (35, 44, 29, 38)      # harness 认的"答案区"
B = (5, 14, 19, 28)       # harness 认的"题面"
for nm, (r0, r1, c0, c1) in {"答案区 A": A, "题面 B": B}.items():
    sub = g[r0:r1+1, c0:c1+1]
    print(f"\n{nm} {(r0,r1,c0,c1)}  颜色 {sorted(set(sub.flatten().tolist()))}")
    for row in sub:
        print("   " + "".join(PAL[v % 16] for v in row))
d = (g[A[0]:A[1]+1, A[2]:A[3]+1] != g[B[0]:B[1]+1, B[2]:B[3]+1])
print(f"\n两区像素差 {int(d.sum())} 格")
print("差异位置:", np.argwhere(d).tolist()[:12])
