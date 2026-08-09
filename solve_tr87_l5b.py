"""L5 结构化求解: 每对字典调成与某例句映射一致(模变换), 4 对组合 clone 验证。"""
import copy, itertools, json, time
import numpy as np
import arc_agi
from arcengine import GameAction, ActionInput

ACTS = {i: getattr(GameAction, f"ACTION{i}") for i in range(1, 8)}
OPS = {"r0": lambda x: x, "r90": lambda x: np.rot90(x, -1), "r180": lambda x: np.rot90(x, 2),
       "r270": lambda x: np.rot90(x, 1), "fx": np.fliplr, "fy": np.flipud,
       "ft": lambda x: x.T, "fa": lambda x: np.rot90(x.T, 2)}
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
g0 = np.array(f.frame[-1])

def glyphs(g, r0, c0, c1):
    out = []
    c = c0
    while c + 4 <= c1:
        out.append(np.array([[1 if g[r0 + i][c + j] == 5 else 0 for j in range(5)] for i in range(5)]))
        c += 7
    return out

# 例句
exA = glyphs(g0, 44, 15, 47)
exS = glyphs(g0, 53, 15, 48)
maps = list(zip(exA, exS))

BOX = [(10, 16, 8, 14), (10, 16, 18, 24), (10, 16, 31, 37), (10, 16, 41, 55),
       (22, 28, 8, 21), (22, 28, 25, 31), (22, 28, 38, 44), (22, 28, 48, 54)]

def box_glyphs(g, b):
    r0, r1, c0, c1 = b
    return glyphs(g, r0 + 1, c0 + 1, c1 - 1)

# 每框环内容
rings = []
for i in range(8):
    ch = clone(game)
    for _ in range(i):
        raw(ch, 4)
    fr = raw(ch, 3); fr = raw(ch, 4)
    seq_ = [box_glyphs(np.array(fr.frame[-1]), BOX[i])]
    for _ in range(8):
        fr = raw(ch, 1)
        cc = box_glyphs(np.array(fr.frame[-1]), BOX[i])
        if all(np.array_equal(a, b) for a, b in zip(cc, seq_[0])) and len(cc) == len(seq_[0]):
            break
        seq_.append(cc)
    rings.append(seq_)

# 每对可行 (kA, kB): 存在 m,T 使 T(m.A) ∈ A框符号串 且 T(m.S) ∈ B框符号串
pair_opts = []
for p in range(4):
    ra, rb = rings[2 * p], rings[2 * p + 1]
    opts = []
    for kA, kB in itertools.product(range(len(ra)), range(len(rb))):
        for mi, (ma, ms) in enumerate(maps):
            okA = any(np.array_equal(x, fn(ma)) for fn in OPS.values() for x in ra[kA])
            okS = any(np.array_equal(x, fn(ms)) for fn in OPS.values() for x in rb[kB])
            if okA and okS:
                opts.append((kA, kB, mi, "any"))
                break
    pair_opts.append(opts)
    print(f"对{p+1}: {len(opts)} 可行 (kA,kB) -> {[(a, b, m, o) for a, b, m, o in opts[:8]]}", flush=True)

def keyseq(ks):
    seq = []
    nz = [i for i, k in enumerate(ks) if k > 0]
    if not nz:
        return []
    last = nz[-1]
    for i in range(8):
        k = ks[i]
        n = len(rings[i])
        if k > 0:
            seq += [1] * k if k <= n - k else [2] * (n - k)
        if i < last:
            seq.append(4)
    return seq

t0 = time.time(); tried = 0; found = None
for combo in itertools.product(*pair_opts):
    ks = [0] * 8
    for p, (kA, kB, mi, op) in enumerate(combo):
        ks[2 * p], ks[2 * p + 1] = kA, kB
    seq = keyseq(ks)
    ch = clone(game)
    win = False
    for a in seq:
        fr = raw(ch, a)
        if fr.levels_completed >= level:
            win = True; break
    tried += 1
    if win:
        found = seq
        print(f"命中! combo={[(c[0],c[1]) for c in combo]} maps={[c[2] for c in combo]} {len(seq)}步 ({tried}试, {time.time()-t0:.0f}s)", flush=True)
        break
print(f"尝试 {tried} 组合 ({time.time()-t0:.0f}s)")
if found:
    for a in found:
        f = env.step(ACTS[a])
    print(f"真机: levels={f.levels_completed} ({len(found)}步 vs 人类71)")
    if f.levels_completed >= level:
        sols["seqs"].append(found)
        json.dump(sols, open("tr87_solutions.json", "w"))
        print("已存")
