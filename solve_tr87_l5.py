"""L5: 8 个字典框, 汉明距离递增穷举(多数框应已正确)。"""
import copy, itertools, json, time
import numpy as np
import arc_agi
from arcengine import GameAction, ActionInput

ACTS = {i: getattr(GameAction, f"ACTION{i}") for i in range(1, 8)}
def raw(g, a): return g.perform_action(ActionInput(id=ACTS[a], data={}), raw=True)
def clone(g):
    clean = getattr(g, "_clean_levels", None)
    g._clean_levels = None; g2 = copy.deepcopy(g)
    g._clean_levels = clean; g2._clean_levels = clean
    return g2

arc = arc_agi.Arcade()
env = arc.make("tr87")
f = env.reset()
sols = json.load(open("tr87_solutions.json"))
for seq in sols["seqs"]:
    for a in seq:
        f = env.step(ACTS[a])
game = env._game
level = f.levels_completed + 1

# 8 框: 光标顺序确定的区域(行, 列)已知; 直接用"整帧减光标行"做符号签名太粗,
# 用每框固定 bbox: 由光标探测得 (r0..r1, c0..c1) 内缩
BOXES = [(10, 16, 9, 13), (10, 16, 19, 23), (10, 16, 32, 36), (10, 16, 42, 53),
         (22, 28, 9, 20), (22, 28, 26, 30), (22, 28, 39, 43), (22, 28, 49, 53)]

def sig(g, b):
    r0, r1, c0, c1 = b
    return tuple(map(tuple, g[r0 + 1:r1, c0:c1 + 1]))

# 每框环长(在 clone 上: 移到框 i, 按 ACTION1 循环)
rings_len = []
for i in range(8):
    ch = clone(game)
    for _ in range(i):
        raw(ch, 4)
    g = np.array(f.frame[-1]) if i == 0 else np.array(raw(ch, 7).frame[-1] if False else ch.frames[-1].frame[-1]) if False else None
    # 直接从 clone 当前帧取: 走一步空? 用 raw(ch,3)+raw(ch,4) 回到原位取帧
    fr = raw(ch, 3); fr = raw(ch, 4)  # 左右回位, 光标不变但拿到帧
    for _ in range(i):
        pass
    g0 = np.array(fr.frame[-1])
    s0 = sig(g0, BOXES[i])
    n = 0
    for _ in range(15):
        fr = raw(ch, 1)
        n += 1
        if sig(np.array(fr.frame[-1]), BOXES[i]) == s0:
            break
    rings_len.append(n)
print(f"各框环长: {rings_len}", flush=True)

R = max(rings_len)
t0 = time.time()
found = None
for kk in range(4, 6):
    cnt = 0
    for pos in itertools.combinations(range(8), kk):
        for vals in itertools.product(*[range(1, rings_len[i]) for i in pos]):
            ks = [0] * 8
            for i, v in zip(pos, vals):
                ks[i] = v
            seq = []
            last = max(i for i in pos)
            for i in range(8):
                n = rings_len[i]
                k = ks[i]
                if k > 0:
                    seq += [1] * k if k <= n - k else [2] * (n - k)
                if i < last:
                    seq.append(4)
            ch = clone(game)
            win = False
            for a in seq:
                fr = raw(ch, a)
                if fr.levels_completed >= level:
                    win = True; break
            cnt += 1
            if win:
                found = seq
                print(f"命中! 改{kk}框 pos={pos} vals={vals} {len(seq)}步 ({time.time()-t0:.0f}s)", flush=True)
                break
        if found: break
    print(f"k={kk}: {cnt} 组合, 累计 {time.time()-t0:.0f}s", flush=True)
    if found: break

if found:
    for a in found:
        f = env.step(ACTS[a])
    print(f"真机: levels={f.levels_completed} ({len(found)}步 vs 人类71)")
    if f.levels_completed >= level:
        sols["seqs"].append(found)
        json.dump(sols, open("tr87_solutions.json", "w"))
        print("已存")
