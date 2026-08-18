"""L3: 方块只能停在有限个格点上(四向各移 4 格)。枚举每个位置 x 画布全2, 查过关。

已知: 方块四向可达边界 行11.5-31.5 x 列28.5-44.5 -> 6 x 5 = 30 个格点。
画布(九宫格)中间列是空(色0), 点一下变色2 -> h1=0 可达。
判据只认 level 上升(含补步)。
"""
import json
import numpy as np
from harness.env import Action, Game
from harness.run import _parse

sol = json.load(open("sc25_solutions.json"))
game, obs = Game.make("sc25")
for t in sol["seq"]:
    obs = game.act(_parse(t))
L0 = obs.level
R, C = [49, 54, 59], [24, 29, 34]
MID = [Action.click(C[1] + 1, R[i] + 1) for i in range(3)]   # 中间列三格

def blk(g):
    m = ((g == 9) | (g == 10)); m[37:43, 22:31] = False; m[49:62, 24:37] = False
    c = np.argwhere(m)
    return (round(float(c[:, 0].mean()), 1), round(float(c[:, 1].mean()), 1)) if len(c) else None

def canvas3(g):
    return [[int(g[r + 1, c + 1]) for c in C] for r in R]

hits = []
tried = 0
for dn in range(0, 6):            # 下移次数
    for lf in range(-3, 4):       # 负=右移, 正=左移
        n = game.fork(); ok = True
        for _ in range(dn):
            if n.act(Action.key(2)).dead: ok = False; break
        if ok:
            for _ in range(abs(lf)):
                if n.act(Action.key(3 if lf > 0 else 4)).dead: ok = False; break
        if not ok:
            continue
        pos = blk(np.array(n._grid()))
        # 画布点成全 2: 中间列三格各点一次(顺序无关, 点完补一步结算)
        o = None
        won = None
        for a in MID + [MID[0]]:
            o = n.act(a)
            if o.dead: break
            if o.level > L0: won = "直接"; break
            p = n.fork().act(MID[0])
            if not p.dead and p.level > L0: won = "补步"; break
        tried += 1
        if won:
            hits.append((pos, dn, lf, won, canvas3(np.array(o.grid))))
            print(f"🏆 过关! 方块 {pos} (下{dn} 左{lf}) | {won}")
print(f"\n枚举 {tried} 个方块位置 x 画布全2: 过关 {len(hits)} 个")
if not hits:
    print("=> **画布全2 + 方块任意位置都不过关** —— 过关条件不在这两个维度里")
