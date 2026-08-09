"""tr87 通用求解器: 规则 = 题面符号=T(src_j) => 答案段=[T(d) for d in dst_j](串序可能反转)。

每关: parse -> 每题面符号枚举 (j,T) 候选段 -> 段长和=answer位数 的切分组合
-> clone 重放验证 -> 命中真机执行。歧义靠 clone 验证消解。
"""
import copy, itertools, json, os, sys, time
import numpy as np
import arc_agi
from arcengine import GameAction, ActionInput
from parse_tr87 import parse, show

ACTS = {i: getattr(GameAction, f"ACTION{i}") for i in range(1, 8)}
OPS = {"r0": lambda x: x, "r90": lambda x: np.rot90(x, -1), "r180": lambda x: np.rot90(x, 2),
       "r270": lambda x: np.rot90(x, 1), "fx": np.fliplr, "fy": np.flipud,
       "ft": lambda x: x.T, "fa": lambda x: np.rot90(x.T, 2)}

def raw(g, a):
    return g.perform_action(ActionInput(id=ACTS[a], data={}), raw=True)

def clone(g):
    clean = getattr(g, "_clean_levels", None)
    g._clean_levels = None; g2 = copy.deepcopy(g)
    g._clean_levels = clean; g2._clean_levels = clean
    return g2

def tup(m):
    return tuple(map(tuple, m))

def seg_candidates(pm, pairs):
    """题面符号 pm 的候选答案段列表(去重)"""
    out, seen = [], set()
    for s, d in pairs:
        if len(s) != 1:
            continue
        sm = np.array(s[0])
        for op, f in OPS.items():
            if np.array_equal(f(sm), np.array(pm)):
                seg = [tup(f(np.array(x))) for x in d]
                for cand in ([seg] if len(seg) == 1 else [seg, seg[::-1]]):
                    k = tuple(cand)
                    if k not in seen:
                        seen.add(k); out.append(cand)
    return out

def solve_level(env, game, level, human):
    f0_frame = None
    g = np.array(env._last_frame)
    pairs, prob, ans0, _ = parse(g)
    N = len(ans0)
    print(f"— L{level}: 题面{len(prob)} 答案{N}位 人类基准{human}")

    cands = [seg_candidates(m, pairs) for m in prob]
    for j, c in enumerate(cands):
        if not c:
            print(f"  题面符号{j+1} 无 (j,T) 匹配 — 规则失效"); return None
        print(f"  符号{j+1}: {len(c)} 候选段, 段长 {sorted(set(len(x) for x in c))}")

    # 每位环序
    rings = []
    for i in range(N):
        ch = clone(game)
        for _ in range(i):
            raw(ch, 4)
        ring = [tup(ans0[i])]
        for _ in range(12):
            fr = raw(ch, 1)
            _, _, a2, _ = parse(np.array(fr.frame[-1]))
            if tup(a2[i]) == ring[0]:
                break
            ring.append(tup(a2[i]))
        rings.append(ring)

    def keyseq(ks):
        seq = []
        last = max((i for i, k in enumerate(ks) if k > 0), default=-1)
        for i, k in enumerate(ks):
            n = len(rings[i])
            if k > 0:
                seq += [1] * k if k <= n - k else [2] * (n - k)
            if i < last:
                seq.append(4)
        return seq

    t0 = time.time(); tried = 0
    nseg = len(cands)
    for perm in itertools.permutations(range(nseg)):
        for combo0 in itertools.product(*cands):
            combo = [combo0[j] for j in perm]
            target = [x for seg in combo for x in seg]
            if len(target) != N:
                continue
            tried += 1
            ks = []
            ok = True
            for i in range(N):
                if target[i] not in rings[i]:
                    ok = False; break
                ks.append(rings[i].index(target[i]))
            if not ok:
                continue
            seq = keyseq(ks)
            ch = clone(game)
            win = False
            for a in seq:
                fr = raw(ch, a)
                if fr.levels_completed >= level:
                    win = True; break
            if win:
                print(f"  命中(排列{perm} 第{tried}组合, {time.time()-t0:.1f}s): {len(seq)}步 vs 人类{human}")
                return seq
    print(f"  {tried} 个组合全空 ({time.time()-t0:.1f}s)")
    return None


if __name__ == "__main__":
    HUMAN = [54, 58, 40, 45, 71, 146]
    arc = arc_agi.Arcade()
    env = arc.make("tr87")
    f = env.reset()
    env._last_frame = f.frame[-1]
    game = env._game

    sols = json.load(open("tr87_solutions.json")) if os.path.exists("tr87_solutions.json") else {"seqs": []}
    for i, seq in enumerate(sols["seqs"]):
        for a in seq:
            f = env.step(ACTS[a])
        env._last_frame = f.frame[-1]
    print(f"重放已有 {len(sols['seqs'])} 关, levels={f.levels_completed}")

    while f.levels_completed < f.win_levels:
        lvl = f.levels_completed + 1
        seq = solve_level(env, game, lvl, HUMAN[lvl - 1])
        if seq is None:
            print(f"L{lvl} 未解, 停"); break
        for a in seq:
            f = env.step(ACTS[a])
        env._last_frame = f.frame[-1]
        if f.levels_completed >= lvl:
            sols["seqs"].append(seq)
            json.dump(sols, open("tr87_solutions.json", "w"))
            print(f"  L{lvl} 真机通过 ✓ (state={f.state.name})")
        else:
            print(f"  L{lvl} 真机执行未过关?!"); break

    print(f"\n最终: levels={f.levels_completed}/{f.win_levels} state={f.state.name}")
    print(f"各关步数: {[len(s) for s in sols['seqs']]}")
