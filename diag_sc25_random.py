"""L3: 我读不准画面(滞后骗了四次), 那就**别读画面** —— 纯行为找过关条件。

大量确定性随机序列, **只看 level**。回答两个我一直没数据的问题:
  ① L3 存不存在"能过关的行为"(哪怕撞运气)
  ② 若撞到, 过关序列长什么样(动作构成/长度)
不依赖我对画面的任何解读。
"""
import json
import random
import numpy as np
from harness.env import Action, Game, action_space
from harness.percept import analyze
from harness.run import _parse

sol = json.load(open("sc25_solutions.json"))
game, obs = Game.make("sc25")
for t in sol["seq"]:
    obs = game.act(_parse(t))
L0 = obs.level
sp = action_space(list(obs.actions))
sc = analyze(obs.grid)
ACTS = [Action.key(k) for k in sp["keys"]] + [Action.click(c, r) for (r, c) in sc.targets]
print(f"L{L0+1}: 动作 {len(ACTS)} 个; 随机序列, 只看 level")

rng = random.Random(20260818)
hits = []
TRIALS, LEN = 4000, 40
for t in range(TRIALS):
    n = game.fork()
    seq = []
    for _ in range(LEN):
        a = ACTS[rng.randrange(len(ACTS))]
        o = n.act(a)
        seq.append(a)
        if o.dead:
            break
        if o.level > L0:
            hits.append(list(seq))
            break
    if (t + 1) % 1000 == 0:
        print(f"  试了 {t+1} 条, 过关 {len(hits)} 条", flush=True)
print(f"\n{TRIALS} 条随机序列(每条最多 {LEN} 步): **过关 {len(hits)} 条**")
if hits:
    hits.sort(key=len)
    best = hits[0]
    nk = sum(1 for a in best if a.id <= 5)
    print(f"最短过关序列 {len(best)} 步 (按键 {nk} / 点击 {len(best)-nk}):")
    print("  ", [str(a) for a in best])
    json.dump({"game": "sc25", "level": 3, "seq": [str(a) for a in best]},
              open("sc25_l3_random_hit.json", "w"), ensure_ascii=False, indent=1)
else:
    print("=> 4000 条随机序列一条都没撞中 —— 过关条件很窄, 盲搜/随机都不可行")
