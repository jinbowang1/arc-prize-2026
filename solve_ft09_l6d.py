"""L6: 汉明球搜索(want态/全B态附近), lights-out 逆解 + clone 验证。"""
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

WANT_E = [(16,6),(16,14),(16,38),(24,22),(24,30),(24,46),(32,14),(32,30),(32,38),(40,22),(40,46),(40,54)]
base_need = [0]*n
for p in WANT_E:
    base_need[idx[p]] = 1

def clicks_for(need):
    out = []
    for x in xs:
        col = sorted([y for (y, xx) in cells if xx == x], reverse=True)
        flip = {y: 0 for y in col}
        for y in col:
            if (need[idx[(y, x)]] + flip[y]) % 2 == 1:
                out.append((x, y))
                flip[y] += 1
                if (y - 8, x) in flip:
                    flip[y - 8] += 1
    return out

def verify(clicks):
    if not clicks:
        return False
    ch = clone(game)
    for (x, y) in clicks:
        fr = raw(ch, 6, {"x": x, "y": y})
        if fr.levels_completed >= level:
            return True
    return False

t0 = time.time()
tested = 0
union_need = [0]*n
for p in [(16,6),(16,14),(16,38),(24,30),(24,46),(32,30),(32,38),(24,22),(32,14),(40,22),(40,46),(40,54)]:
    union_need[idx[p]] = 1
for center_name, center in (("want", base_need), ("allB", [0]*n), ("union", union_need)):
    for k in range(0, 5):
        for extra in itertools.combinations(range(n), k):
            need = list(center)
            for i in extra:
                need[i] ^= 1
            cl = clicks_for(need)
            tested += 1
            if verify(cl):
                print(f"命中! 中心={center_name} 翻转差={[cells[i] for i in extra]} {len(cl)}击 ({tested}试 {time.time()-t0:.0f}s)")
                for (x, y) in cl:
                    f = env.step(GameAction.ACTION6, {"x": x, "y": y})
                print(f"真机 levels={f.levels_completed} state={f.state.name}")
                if f.levels_completed >= level:
                    sols["seqs"].append(cl)
                    json.dump(sols, open("ft09_solutions.json", "w"))
                    print("已存 — ft09 全通!" if f.levels_completed >= 6 else "已存")
                raise SystemExit
    print(f"{center_name} k<=4 空 ({tested}试 {time.time()-t0:.0f}s)", flush=True)
print("全空")
