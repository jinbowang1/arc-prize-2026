"""L6: allE 汉明球 + 计数型判定全枚举。"""
import itertools, json, time
import numpy as np
import arc_agi
from arcengine import GameAction, ActionInput
from solve_ft09_all import blocks, clone, raw

arc = arc_agi.Arcade()
env = arc.make("ft09")
f = env.reset()
sols = json.load(open("ft09_solutions.json"))
for seq in sols["seqs"]:
    for (x, y) in seq:
        f = env.step(GameAction.ACTION6, {"x": x, "y": y})
game = env._game
g0 = np.array(f.frame[-1])
level = f.levels_completed + 1

cands = blocks(g0)
cells = []
for (y, x) in cands:
    ch = clone(game)
    fr = raw(ch, 6, {"x": x, "y": y})
    if np.any(np.array(fr.frame[-1])[:63] != g0[:63]):
        cells.append((y, x))
cells.sort()
idx = {p: i for i, p in enumerate(cells)}
n = len(cells)
xs = sorted({x for _, x in cells})

def clicks_for(need):
    out = []
    for x in xs:
        col = sorted([y for (y, xx) in cells if xx == x], reverse=True)
        flip = {y: 0 for y in col}
        for y in col:
            if (need[idx[(y, x)]] + flip[y]) % 2 == 1:
                out.append((x, y)); flip[y] += 1
                if (y - 8, x) in flip:
                    flip[y - 8] += 1
    return out

def verify(cl):
    if not cl:
        return False
    ch = clone(game)
    for (x, y) in cl:
        fr = raw(ch, 6, {"x": x, "y": y})
        if fr.levels_completed >= level:
            return True
    return False

def finish(cl, tag):
    print(f"命中! {tag} {len(cl)}击")
    for (x, y) in cl:
        globals()['f'] = env.step(GameAction.ACTION6, {"x": x, "y": y})
    fx = globals()['f']
    print(f"真机 levels={fx.levels_completed} state={fx.state.name}")
    if fx.levels_completed >= level:
        sols["seqs"].append(cl)
        json.dump(sols, open("ft09_solutions.json", "w"))
        print("已存 — ft09 全通!" if fx.levels_completed >= 6 else "已存")
    raise SystemExit

t0 = time.time()
# 1) allE + k<=4 球
tested = 0
for k in range(0, 5):
    for extra in itertools.combinations(range(n), k):
        need = [1] * n
        for i in extra:
            need[i] ^= 1
        cl = clicks_for(need)
        tested += 1
        if verify(cl):
            finish(cl, f"allE 差{k}")
print(f"allE k<=4 空 ({tested}试 {time.time()-t0:.0f}s)", flush=True)

# 2) 计数型: 每花纹邻域 E 数 = 蓝图 0 位数
PATS = [((8,14), [(8,6),(16,6),(16,14),(16,22)], 2),
        ((24,38), [(16,30),(16,38),(16,46),(24,30),(24,46),(32,30),(32,38),(32,46)], 5),
        ((32,22), [(24,14),(24,22),(24,30),(32,14),(32,30),(40,14),(40,22),(40,30)], 5),
        ((48,46), [(40,38),(40,46),(40,54),(48,54)], 2)]
# DFS 枚举满足全部计数约束的态
neigh_sets = [set(p[1]) for p in PATS]
quotas = [p[2] for p in PATS]
free = [p for p in cells if not any(p in s for s in neigh_sets)]
assert not free, f"未覆盖 {free}"

count = 0
def gen(states, i):
    global count
    if i == n:
        if all(sum(states[idx[q]] for q in s) == q0 for s, q0 in zip(neigh_sets, quotas)):
            yield list(states)
        return
    p = cells[i]
    for v in (0, 1):
        states.append(v)
        # 剪枝: 已满配额检查
        ok = True
        for s, q0 in zip(neigh_sets, quotas):
            done = [q for q in s if idx[q] <= i]
            cur = sum(states[idx[q]] for q in done)
            if cur > q0 or cur + (len(s) - len(done)) < q0:
                ok = False; break
        if ok:
            yield from gen(states, i + 1)
        states.pop()

tested2 = 0
for need in gen([], 0):
    cl = clicks_for(need)
    tested2 += 1
    if verify(cl):
        finish(cl, f"计数型态#{tested2}")
    if tested2 % 2000 == 0:
        print(f"  计数枚举 {tested2} ({time.time()-t0:.0f}s)", flush=True)
print(f"计数型全空 ({tested2}态 {time.time()-t0:.0f}s)")
