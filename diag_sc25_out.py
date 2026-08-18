"""L3: 门是死墙(98 组合零反应)。回到 L1 的模式试: 把方块推出管道?
四个方向各推到底, 看方块会不会消失/过关。"""
import json
import numpy as np
from harness.env import Action, Game
from harness.run import _parse

sol = json.load(open("sc25_solutions.json"))
game, obs = Game.make("sc25")
for t in sol["seq"]:
    obs = game.act(_parse(t))
L0 = obs.level

def blk(g):
    m = ((g == 9) | (g == 10)); m[37:43, 22:31] = False; m[49:62, 24:37] = False
    c = np.argwhere(m)
    return (round(float(c[:, 0].mean()), 1), round(float(c[:, 1].mean()), 1)) if len(c) else None

for name, k in {"上 A1": 1, "下 A2": 2, "左 A3": 3, "右 A4": 4}.items():
    n = game.fork(); path = []; won = None
    for i in range(12):
        o = n.act(Action.key(k))
        if o.dead: path.append("DEAD"); break
        if o.level > L0: won = i + 1; break
        p = n.fork().act(Action.key(k))          # 补步查过关
        if not p.dead and p.level > L0: won = f"{i+1}(补步)"; break
        path.append(str(blk(np.array(o.grid))))
    tail = f" 🏆过关于第 {won} 步" if won else ""
    # 只显示位置变化的点
    uniq = []
    for x in path:
        if not uniq or uniq[-1] != x: uniq.append(x)
    print(f"{name}: {' '.join(uniq)}{tail}")
