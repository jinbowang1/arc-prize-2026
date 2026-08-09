"""tr87 等价类求解器: 一切匹配在 canonical form(模旋转/镜像)层面。

用户(人类通关者)的洞察: 判定与朝向无关, 只认形状本身。
预期: L1-L4 从克隆树穷举(9s~9.5min)退化为查表 + 少量组合验证。
"""
import copy, itertools, json, time
import numpy as np
import arc_agi
from arcengine import GameAction, ActionInput
from parse_tr87 import parse, show

ACTS = {i: getattr(GameAction, f"ACTION{i}") for i in range(1, 8)}
OPS = [lambda x: x, lambda x: np.rot90(x, -1), lambda x: np.rot90(x, 2), lambda x: np.rot90(x, 1),
       np.fliplr, np.flipud, lambda x: np.array(x).T, lambda x: np.rot90(np.array(x).T, 2)]

def raw(g, a):
    return g.perform_action(ActionInput(id=ACTS[a], data={}), raw=True)

def clone(g):
    clean = getattr(g, "_clean_levels", None)
    g._clean_levels = None; g2 = copy.deepcopy(g)
    g._clean_levels = clean; g2._clean_levels = clean
    return g2

def can(m):
    m = np.asarray(m)
    return min(tuple(map(tuple, f(m).tolist())) for f in OPS)

def get_rings(game, N):
    rings = []
    for i in range(N):
        ch = clone(game)
        for _ in range(i):
            raw(ch, 4)
        ring = None
        for _ in range(13):
            fr = raw(ch, 1)
            _, _, a2, _ = parse(np.array(fr.frame[-1]))
            c = can(a2[i])
            if ring is None:
                ring = []
            if ring and c == ring[0][1]:
                break
            ring.append((len(ring) + 1, c))   # (拨动次数, 类)
        rings.append(ring)
    return rings

def solve_translate(env, game, level, human):
    """L1-L4 翻译关: 等价类查表 + 切分/排列组合验证。"""
    t0 = time.time()
    g = np.array(env._last_frame)
    pairs, prob, ans0, _ = parse(g)
    N = len(ans0)
    LUT = {}
    for s, d in pairs:
        if s and d:
            LUT[tuple(can(x) for x in s)] = [can(x) for x in d]
    if not LUT:
        # 无杠框阵布局(L4): 扫描框顶行, 带内相邻两框配对, 建单符号链式 LUT
        from parse_tr87 import runs_in_row, glyphs_in_box, nonempty
        bg = 2
        r = 0
        pairs2 = []
        while r < 36:
            rs = [x for x in runs_in_row(g[r], bg) if x[1] - x[0] >= 6]
            if len(rs) >= 2 and (r == 0 or not [x for x in runs_in_row(g[r - 1], bg) if x[1] - x[0] >= 6]):
                for i in range(0, len(rs) - 1, 2):
                    a0, a1, _ = rs[i]; b0, b1, _ = rs[i + 1]
                    s = nonempty(glyphs_in_box(g, r, a0, a1))
                    d = nonempty(glyphs_in_box(g, r, b0, b1))
                    if s and d:
                        pairs2.append((s, d))
                r += 7
            else:
                r += 1
        flat = {}
        for s, d in pairs2:
            if len(s) == 1 and len(d) == 1:
                flat[can(s[0])] = can(d[0])
        # 链式复合: 值若还能继续查表则往下走(最多2跳), 终点为答案语言
        for k, v in flat.items():
            LUT[(k,)] = [flat.get(v, v)]
        print(f"  L{level} 框阵字典: {len(pairs2)} 对, 链式 LUT {len(LUT)} 键", flush=True)
    prob_c = [can(m) for m in prob]
    ans0_c = [can(m) for m in ans0]

    # 切分: 题面 can 序列被 LUT 键(长1-3 窗口)覆盖的所有方案
    P = len(prob_c)
    def cuts(i):
        if i == P:
            yield []
            return
        for key, val in LUT.items():
            L = len(key)
            if tuple(prob_c[i:i + L]) == key:
                for rest in cuts(i + L):
                    yield [list(val)] + rest
    rings = get_rings(game, N)
    ring_lut = [{c: k for k, c in reversed(r)} for r in rings]   # 类->最小拨动次数(取首现)

    tried = 0
    for segs in cuts(0):
        for perm in itertools.permutations(range(len(segs))):
            for revs in itertools.product((0, 1), repeat=len(segs)):
                target = []
                for si in perm:
                    seg = segs[si][::-1] if revs[si] else segs[si]
                    target += seg
                if len(target) != N:
                    continue
                ks = []
                ok = True
                for i in range(N):
                    if target[i] == ans0_c[i]:
                        ks.append(0)
                    elif target[i] in ring_lut[i]:
                        ks.append(ring_lut[i][target[i]])
                    else:
                        ok = False; break
                if not ok or all(k == 0 for k in ks):
                    continue
                tried += 1
                seq = []
                last = max(i for i, k in enumerate(ks) if k > 0)
                for i, k in enumerate(ks):
                    if k > 0:
                        seq += [1] * k if k <= 7 - k else [2] * (7 - k)
                    if i < last:
                        seq.append(4)
                ch = clone(game)
                win = False
                for a in seq:
                    fr = raw(ch, a)
                    if fr.levels_completed >= level:
                        win = True; break
                if win:
                    dt = time.time() - t0
                    print(f"  L{level} 命中: {len(seq)}步 vs 人类{human} (验证{tried}个组合, {dt:.1f}s)", flush=True)
                    return seq
    print(f"  L{level} 未解(验证{tried})", flush=True)
    return None

HUMAN = [54, 58, 40, 45, 71, 146]
arc = arc_agi.Arcade()
env = arc.make("tr87")
f = env.reset()
env._last_frame = f.frame[-1]
game = env._game
new_sols = {"seqs": []}
old = json.load(open("tr87_solutions.json"))

T0 = time.time()
while f.levels_completed < f.win_levels:
    lvl = f.levels_completed + 1
    if lvl <= 4:
        seq = solve_translate(env, game, lvl, HUMAN[lvl - 1])
        if seq is None:
            break
    else:
        seq = old["seqs"][lvl - 1]   # L5/L6 已是等价类方法, 复用
        print(f"  L{lvl}: 复用既有解({len(seq)}步, 原方法已是等价类)", flush=True)
    for a in seq:
        f = env.step(ACTS[a])
    env._last_frame = f.frame[-1]
    assert f.levels_completed >= lvl, f"L{lvl} 执行未过!"
    new_sols["seqs"].append(seq)

print(f"\n最终 {f.levels_completed}/{f.win_levels} state={f.state.name} 总耗时 {time.time()-T0:.0f}s")
print(f"各关步数: {[len(s) for s in new_sols['seqs']]} (旧: {[len(s) for s in old['seqs']]})")
json.dump(new_sols, open("tr87_solutions_canon.json", "w"))
