"""L3: "颜色转移依赖构型"证据不足 —— 那组数据里构型和格子初始色**同时**变了。
干净做法: 固定构型, 只连点同一个格子, 看颜色序列。
再换一个构型, 重复同一串, 比对。两条序列相同 => 不依赖构型。
"""
import json
import numpy as np
from harness.env import Action, Game
from harness.run import _parse

sol = json.load(open("sc25_solutions.json"))
game, obs = Game.make("sc25")
for t in sol["seq"]:
    obs = game.act(_parse(t))
R, C = [49, 54, 59], [24, 29, 34]
cv = lambda g, i, j: int(np.array(g)[R[i] + 1, C[j] + 1])
click = lambda i, j: Action.click(C[j] + 1, R[i] + 1)

def seq_colors(lead_keys, i, j, n=6):
    """构型由 lead_keys 决定; 然后连点格(i,j) n 次, 每次记颜色(读的是上一次的效果)"""
    nd = game.fork()
    for k in lead_keys:
        if nd.act(Action.key(k)).dead:
            return None
    out = []
    for _ in range(n):
        o = nd.act(click(i, j))
        if o.dead:
            out.append("D"); break
        out.append(cv(o.grid, i, j))
    return out

print("格(0,0) 初始色 2 | 格(0,1) 初始色 0\n")
for label, lead in {"构型A(不动)": [], "构型B(下2)": [2, 2], "构型C(左3)": [3, 3, 3],
                    "构型D(下2左3)": [2, 2, 3, 3, 3]}.items():
    a = seq_colors(lead, 0, 0)
    b = seq_colors(lead, 0, 1)
    print(f"{label:<14} 格(0,0)连点: {a}   格(0,1)连点: {b}")
print("\n判读: 若四行的同一列序列**都相同** => 颜色转移**不依赖构型**,")
print("      两格序列不同只是因为**初始色不同**(0<->2 与 2<->14 是两条独立的切换对)。")
