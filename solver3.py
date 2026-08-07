"""ls20 求解器 v6: 语义状态去重 = (格子, 形状, 能量)。

要点: 状态空间因此有界(约60格×24形状×22能量档≈3万), 搜索不再爆炸。
去重会合并巡逻体相位不同的状态 => 可能漏解, 但绝不会给出无效解
(所有转移都在真模拟器上走出来, 找到的路径按构造即已验证)。
"""
import argparse, copy, json, time
import numpy as np
from arcengine import GameAction, ActionInput
from wm import load_env, shape_bits, energy

ACTS = {a: getattr(GameAction, f"ACTION{a}") for a in (1, 2, 3, 4)}


def step(game, a):
    return game.perform_action(ActionInput(id=ACTS[a], data={}), raw=True)


def clone(game, clean):
    game._clean_levels = None
    g2 = copy.deepcopy(game)
    game._clean_levels = clean
    g2._clean_levels = clean
    return g2


def key_cell(g, last):
    """钥匙: 上2行12下3行9 的5x5块; 多候选时取离上次最近的。"""
    c = []
    for r in range(0, 56):
        for col in range(0, 60):
            w = g[r:r+5, col:col+5]
            if (w[0:2] == 12).all() and (w[2:5] == 9).all():
                c.append((r, col))
    if not c:
        return last
    if last is None:
        return c[0]
    return min(c, key=lambda p: abs(p[0]-last[0]) + abs(p[1]-last[1]))


def movers(g):
    """旋转带内(rows30-45)巡逻体的位置: 决定"踩哪格才旋转"的隐藏相位。
    只取该带内, 避免把全图无关动画也塞进指纹导致状态爆炸。"""
    p = np.argwhere((g == 0) | (g == 1))
    p = p[(p[:, 0] >= 30) & (p[:, 0] <= 45)]
    return (int(p[:, 0].min()), int(p[:, 1].min())) if len(p) else ()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix-file", default="solutions_l4.json")
    ap.add_argument("--maxdepth", type=int, default=70)
    ap.add_argument("--with-movers", action="store_true",
                    help="把移动物体位置也纳入指纹(去重更保守, 不易漏解)")
    args = ap.parse_args()

    game, f = load_env(args.prefix_file)
    base = f.levels_completed
    clean = game._clean_levels
    g = np.array(f.frame[-1])
    ext = (lambda gg: (movers(gg),)) if args.with_movers else (lambda gg: ())
    st0 = (key_cell(g, None), shape_bits(g), energy(g)) + ext(g)
    print(f"L{base+1} 起始语义状态 {st0}", flush=True)

    frontier = [(game, [], st0)]
    seen = {st0}
    shapes_seen = {st0[1]}
    at_lock = set()
    t0 = time.time()
    for depth in range(args.maxdepth):
        nxt = []
        for base_game, path, st in frontier:
            for a in (1, 2, 3, 4):
                child = clone(base_game, clean)
                fr = step(child, a)
                if not fr.frame:
                    continue
                if fr.levels_completed > base:
                    seq = path + [a]
                    print(f"*** 过关! {len(seq)} 步: {','.join(map(str, seq))}", flush=True)
                    json.dump({"level5_seq": seq}, open("l5_seq.json", "w"))
                    return
                gg = np.array(fr.frame[-1])
                s2 = (key_cell(gg, st[0]), shape_bits(gg), energy(gg)) + ext(gg)
                if s2 in seen:
                    continue
                seen.add(s2)
                shapes_seen.add(s2[1])
                if s2[0] == (10, 54):
                    at_lock.add(s2[1])
                nxt.append((child, path + [a], s2))
        frontier = nxt
        print(f"depth {depth+1}: frontier {len(frontier)}, seen {len(seen)}, "
              f"{time.time()-t0:.0f}s", flush=True)
        if not frontier:
            print("搜索穷尽, 无解(在此去重粒度下)", flush=True)
            print(f"见过的形状({len(shapes_seen)}种): {sorted(shapes_seen)}", flush=True)
            print(f"其中 413 出现过: {413 in shapes_seen}", flush=True)
            print(f"曾带到锁前(10,54)的形状: {sorted(at_lock)}", flush=True)
            return


if __name__ == "__main__":
    main()
