"""ft09 通用逐关求解: 自动发现可编辑格 -> 组合穷举 -> 真机执行。"""
import copy, itertools, json, os, time
import numpy as np
from collections import Counter, deque
import arc_agi
from arcengine import GameAction, ActionInput

def raw(g, a, data=None):
    return g.perform_action(ActionInput(id=getattr(GameAction, f"ACTION{a}"), data=data or {}), raw=True)

def clone(g):
    clean = getattr(g, "_clean_levels", None)
    g._clean_levels = None; g2 = copy.deepcopy(g)
    g._clean_levels = clean; g2._clean_levels = clean
    return g2

def blocks(g):
    """同色连通块(面积>=9, 非背景/非底条), 返回质心列表"""
    bg = Counter(g.flatten().tolist()).most_common(1)[0][0]
    seen = np.zeros_like(g, dtype=bool)
    out = []
    for r in range(63):
        for c in range(64):
            if seen[r, c] or g[r, c] == bg:
                continue
            col = g[r, c]
            q = deque([(r, c)]); seen[r, c] = True; pts = []
            while q:
                y, x = q.popleft(); pts.append((y, x))
                for dy, dx in ((0,1),(0,-1),(1,0),(-1,0)):
                    ny, nx = y+dy, x+dx
                    if 0 <= ny < 63 and 0 <= nx < 64 and not seen[ny, nx] and g[ny, nx] == col:
                        seen[ny, nx] = True; q.append((ny, nx))
            if len(pts) >= 9:
                ys = [p[0] for p in pts]; xs = [p[1] for p in pts]
                out.append((sum(ys)//len(pts), sum(xs)//len(pts)))
    return out

def solve_level(game, g0, level):
    cands = blocks(g0)
    print(f"  连通块候选 {len(cands)} 个", flush=True)
    cells = []
    for (y, x) in cands:
        ch = clone(game)
        fr = raw(ch, 6, {"x": x, "y": y})
        if fr.levels_completed >= level:
            print(f"  单点直接过关: ({x},{y})")
            return [(x, y)]
        g1 = np.array(fr.frame[-1])
        if np.any(g1[:63] != g0[:63]):   # 排除底部计数条
            ring = 1
            prev = g1
            for _ in range(6):
                fr = raw(ch, 6, {"x": x, "y": y})
                gg = np.array(fr.frame[-1])
                if np.array_equal(gg[:63], g0[:63]):
                    ring += 1; break
                if np.array_equal(gg[:63], prev[:63]):
                    break
                ring += 1; prev = gg
            cells.append((y, x, ring + 0))
    print(f"  可编辑格 {len(cells)} 个, 环长 {sorted(set(c[2] for c in cells))}", flush=True)
    if not cells or len(cells) > 20:
        return None
    R = max(c[2] for c in cells)
    t0 = time.time()
    combos = sorted(itertools.product(*[range(c[2]) for c in cells]), key=sum)
    for tried, combo in enumerate(combos):
        if sum(combo) == 0:
            continue
        ch = clone(game)
        win = False
        for (y, x, _), k in zip(cells, combo):
            for _ in range(k):
                fr = raw(ch, 6, {"x": x, "y": y})
                if fr.levels_completed >= level:
                    win = True; break
            if win: break
        if win:
            seq = []
            for (y, x, _), k in zip(cells, combo):
                seq += [(x, y)] * k
            print(f"  命中 combo(点击{sum(combo)}次) 试{tried} ({time.time()-t0:.0f}s)", flush=True)
            return seq
        if tried % 2000 == 0 and tried:
            print(f"  ...{tried}/{len(combos)} ({time.time()-t0:.0f}s)", flush=True)
    return None

if __name__ == "__main__":
    HUMAN = [43, 12, 23, 28, 65, 37]
    arc = arc_agi.Arcade()
    env = arc.make("ft09")
    f = env.reset()
    game = env._game
    sols = json.load(open("ft09_solutions.json")) if os.path.exists("ft09_solutions.json") else {"seqs": []}
    for seq in sols["seqs"]:
        for (x, y) in seq:
            f = env.step(GameAction.ACTION6, {"x": x, "y": y})
    print(f"重放 {len(sols['seqs'])} 关, levels={f.levels_completed}", flush=True)

    while f.levels_completed < f.win_levels:
        lvl = f.levels_completed + 1
        print(f"— L{lvl} (人类{HUMAN[lvl-1]})", flush=True)
        g0 = np.array(f.frame[-1])
        seq = solve_level(game, g0, lvl)
        if seq is None:
            print("未解, 停"); break
        for (x, y) in seq:
            f = env.step(GameAction.ACTION6, {"x": x, "y": y})
        if f.levels_completed >= lvl:
            sols["seqs"].append(seq)
            json.dump(sols, open("ft09_solutions.json", "w"))
            print(f"  L{lvl} ✓ {len(seq)}步 state={f.state.name}", flush=True)
        else:
            print("  真机未过?!"); break
    print(f"最终 {f.levels_completed}/{f.win_levels} state={f.state.name} 各关步数={[len(s) for s in sols['seqs']]}")
