"""L3: 穷尽用的动作集只有 10 个点击(28 个目标筛掉 18 个)。
那 18 个"无效"点击, 在**方块移到别处 / 画布涂过之后**会不会变有效?
—— cd82 上栽过同款: 在少数状态上验证的"无效"不等于处处无效。"""
import json
import numpy as np
from harness.env import Action, Game, action_space
from harness.percept import analyze
from harness.run import _parse

sol = json.load(open("sc25_solutions.json"))
game, obs = Game.make("sc25")
for t in sol["seq"]:
    obs = game.act(_parse(t))
g0 = np.array(obs.grid)
sc = analyze(obs.grid)
sp = action_space(list(obs.actions))
clicks = [Action.click(c, r) for (r, c) in sc.targets]
keys = [Action.key(k) for k in sp["keys"]]
game.detect_lag(keys + clicks)
R, C = [49, 54, 59], [24, 29, 34]

# 开局判定: 哪些点击"有效"
live0 = []
for a in clicks:
    o = game.effect(a)
    if not o.dead and not np.array_equal(np.array(o.grid), g0):
        live0.append(repr(a))
dead0 = [a for a in clicks if repr(a) not in live0]
print(f"开局: 点击 {len(clicks)} 个 -> 有效 {len(live0)} / 无效 {len(dead0)}")

# 造多种现场: 方块推到各处 + 画布涂过
sites = {}
for name, ks in {"原位": [], "下2": [2,2], "左3": [3,3,3], "上2": [1,1], "右2": [4,4],
                 "下2左3": [2,2,3,3,3]}.items():
    n = game.fork(); ok = True
    for k in ks:
        if n.act(Action.key(k)).dead: ok = False; break
    if ok: sites[name] = n
# 再加两个"画布涂过"的现场
n = game.fork()
for i in range(3):
    n.act(Action.click(C[1] + 1, R[i] + 1))
sites["画布涂过"] = n

print(f"\n那 {len(dead0)} 个开局无效的点击, 在别的现场变有效了吗:")
revived = {}
for name, nd in sites.items():
    base = np.array(nd._grid())
    hits = []
    for a in dead0:
        o = nd.effect(a)
        if not o.dead and not np.array_equal(np.array(o.grid), base):
            hits.append(repr(a))
            revived.setdefault(repr(a), []).append(name)
    print(f"  {name:<10} 变有效的: {len(hits)} 个 {hits[:5]}")
print(f"\n合计: {len(revived)} 个开局无效的点击在某个现场变有效 -> {list(revived)[:8]}")
if revived:
    print("🚨=> 动作集被筛窄了, 穷尽的'无解'结论**不成立**, 要用全集重跑。")
else:
    print("=> 那 18 个确实处处无效, 动作集没问题, 穷尽结论站得住。")
