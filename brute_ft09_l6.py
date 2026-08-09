"""L6 终极兜底: 格雷码遍历全部 2^22 态(每步一次点击, 零克隆)。

点击向量组线性无关 -> c 坐标系格雷码 = 每步执行一个点击, 路过所有态各一次。
命中后由步号重构态向量, 求最短点击集真机执行。
"""
import json, time
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
n = len(cells)
print(f"{n} 格, 遍历 2^{n} = {2**n} 态", flush=True)

ch = clone(game)
t0 = time.time()
hit_step = None
N = 2 ** n
for step in range(1, N):
    j = (step & -step).bit_length() - 1   # 格雷码翻转位
    y, x = cells[j]
    fr = raw(ch, 6, {"x": x, "y": y})
    if fr.levels_completed >= level:
        hit_step = step
        print(f"命中! 步 {step} ({time.time()-t0:.0f}s)", flush=True)
        break
    if step % 200000 == 0:
        print(f"  {step}/{N} ({time.time()-t0:.0f}s, {step/(time.time()-t0):.0f}/s)", flush=True)

if hit_step is None:
    print(f"全空 ({time.time()-t0:.0f}s) — 判定不是纯态匹配!")
    raise SystemExit

# 重构: 格雷码第 step 个态的 c 向量 = step ^ (step >> 1)
gray = hit_step ^ (hit_step >> 1)
c = [(gray >> j) & 1 for j in range(n)]
print(f"c 向量(各格点击奇偶) = {c}")
# 由 c 得目标翻转向量 need = Σ c_j * u_j; u_j = self+up
idx = {p: i for i, p in enumerate(cells)}
need = [0] * n
for j, cj in enumerate(c):
    if cj:
        y, x = cells[j]
        need[idx[(y, x)]] ^= 1
        if (y - 8, x) in idx:
            need[idx[(y - 8, x)]] ^= 1
target_E = [cells[i] for i in range(n) if need[i]]
print(f"目标态: {len(target_E)} 格为E: {target_E}")

# 最短点击集 = c 里为 1 的格各点一次(顺序无关)
clicks = [(cells[j][1], cells[j][0]) for j in range(n) if c[j]]
ch2 = clone(game)
win = False
for (x, y) in clicks:
    fr = raw(ch2, 6, {"x": x, "y": y})
    if fr.levels_completed >= level:
        win = True; break
print(f"复核: {len(clicks)}击 {'WIN' if win else '未过?!'}")
if win:
    for (x, y) in clicks:
        f = env.step(GameAction.ACTION6, {"x": x, "y": y})
    print(f"真机 levels={f.levels_completed} state={f.state.name}")
    if f.levels_completed >= level:
        sols["seqs"].append(clicks)
        json.dump(sols, open("ft09_solutions.json", "w"))
        print("已存 — ft09 全通!")
