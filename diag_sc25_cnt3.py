"""L3 唯一剩下的候选: 右上计数器。它从 32 掉到 16 就停住 —— 为什么?

(画布 x 方块) 已穷尽无解, 全屏帧 diff 说除这两者外只有计数器在变。
不预设它是什么, 直接测: 什么动作让它变、变到哪、会不会归零。
"""
import json
import numpy as np
from harness.env import Action, Game, action_space
from harness.percept import analyze
from harness.run import _parse

PAL = " .:-=+*#%@$&XO<>"
sol = json.load(open("sc25_solutions.json"))
game, obs = Game.make("sc25")
for t in sol["seq"]:
    obs = game.act(_parse(t))
g0 = np.array(obs.grid)
R, C = [49, 54, 59], [24, 29, 34]
sp = action_space(list(obs.actions)); sc = analyze(obs.grid)
CNT = (0, 8, 60, 64)
cnt = lambda g: int((np.array(g)[CNT[0]:CNT[1], CNT[2]:CNT[3]] != 0).sum())

print("计数器区 行0-7 列60-63 开局:")
for r in range(8):
    print(f"  {r} " + "".join(PAL[v % 16] for v in g0[r, 60:64]))
print(f"非零 {cnt(g0)} 格\n")

print("各类动作对计数器的影响(单步):")
for label, a in [("按键 A1", Action.key(1)), ("按键 A3", Action.key(3)),
                 ("点九宫格(0,0)", Action.click(C[0]+1, R[0]+1)),
                 ("点九宫格(1,1)", Action.click(C[1]+1, R[1]+1)),
                 ("点按钮 A6(11,55)", Action.click(11, 55))]:
    o = game.fork().act(a)
    if o.dead: print(f"  {label:<18} 致死"); continue
    print(f"  {label:<18} {cnt(g0)} -> {cnt(o.grid)}")

print("\n只按键(不点击)时计数器会不会动:")
n = game.fork()
for k in range(1, 13):
    o = n.act(Action.key([1,2,3,4][k % 4]))
    if o.dead: print(f"  第{k}步死"); break
    if k % 3 == 0:
        print(f"  按键 {k:>2} 次: 计数器 {cnt(o.grid)}")

print("\n交替(点击+按键)时:")
n = game.fork()
for k in range(1, 13):
    a = Action.click(C[0]+1, R[0]+1) if k % 2 else Action.key(3)
    o = n.act(a)
    if o.dead: print(f"  第{k}步死"); break
    if k % 3 == 0:
        print(f"  交替 {k:>2} 步: 计数器 {cnt(o.grid)}")
