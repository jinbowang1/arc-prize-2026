"""诊断: MoverToAnchor 的因果版该记什么参数?

用 ls20 在案解(7关335步)重放, 对每一关回答三个问题:
  ① discover 的实体表里, 钥匙/角色是不是一个带 movers 的实体?
     (= 因果信息能不能可靠认出"谁能动", 替掉两版都错的尺寸猜)
  ② 通关瞬间(赢步之前那帧), 移动过的 blob 落在什么东西上?
     锚点的颜色/形状哪个跨关稳定?
  ③ 实体 cells 与 blob 的对应关系怎么建(重叠判据)?

输出全部事实, 不下结论 —— 结论要能从打印里逐条对出来。
"""
from __future__ import annotations

import json

import numpy as np

from harness.env import Action, Game, action_space
from harness.percept import background, by_color, by_figure, discover


def blob_line(g, bg, b):
    return f"bbox={b.bbox} {b.height}x{b.width} 格数={b.size} 色={sorted(b.colors)}"


def overlap_cells(b, cells: set) -> int:
    r0, r1, c0, c1 = b.bbox
    return sum(1 for (r, c) in cells if r0 <= r <= r1 and c0 <= c <= c1)


def main():
    sol = json.load(open("solutions.json"))
    assert sol["game"] == "ls20"
    seq = [Action.key(i) for i in sol["seq"]]

    game, obs = Game.make("ls20")
    sp = action_space(list(obs.actions) or [1, 2, 3, 4])
    keys = [Action.key(i) for i in sp["keys"]]
    print(f"动作空间 keys={sp['keys']} 总步数={len(seq)} win_levels={obs.win_levels}", flush=True)

    i = 0
    while obs.level < obs.win_levels and i < len(seq):
        lv = obs.level
        start = np.array(obs.grid)
        bg = background(start)

        # ① 实体发现(与 agent.solve_level 同一姿势)
        game.detect_lag(keys)
        ents, _ = discover(lambda a: np.array(game.effect(a).grid), start, keys)
        print(f"\n===== L{lv+1} =====")
        print(f"实体 {len(ents)} 个:")
        for e in ents[:8]:
            print(f"  {e.line()}")

        # ③ 实体 cells ↔ blob 对应
        blobs = list(by_figure(start, bg)) + list(by_color(start, bg))
        seen, uniq = set(), []
        for b in blobs:
            if b.bbox not in seen:
                seen.add(b.bbox)
                uniq.append(b)
        for e in ents[:8]:
            cells = set(e.cells_set)
            hits = [(overlap_cells(b, cells), b) for b in uniq]
            hits = [(n, b) for n, b in hits if n > 0]
            hits.sort(key=lambda x: -x[0])
            for n, b in hits[:3]:
                frac_b = n / b.size
                print(f"  实体{e.bbox} ∩ blob {blob_line(start, bg, b)}"
                      f"  重叠{n}格 (占blob {frac_b:.0%})")

        # 重放到过关, 记逐帧
        frames = [start]
        while obs.level == lv and i < len(seq):
            obs = game.act(seq[i])
            i += 1
            frames.append(np.array(obs.grid))
        if obs.level == lv:
            print("  ⚠️解用完了还没过关, 停")
            break
        # fit 语义: after = 赢步之前那帧
        before, after = frames[0], frames[-2]
        print(f"  本关用了 {len(frames)-1} 步过关")

        # ② 谁动了 + 落在什么上
        bg_a = background(after)

        def _blobs(g, b):
            out = list(by_figure(g, b)) + list(by_color(g, b))
            s2, u2 = set(), []
            for x in out:
                if x.bbox not in s2:
                    s2.add(x.bbox)
                    u2.append(x)
            return u2

        bb = _blobs(before, bg)
        ab = _blobs(after, bg_a)
        causal_movers = []          # (b, a) 因果确认的移动
        for b in bb:
            kb = b.mask_key(before, bg)
            for a in ab:
                if a.mask_key(after, bg_a) != kb or a.center == b.center:
                    continue
                # 这是一个"移动过的同形状块"。它移动前是不是可动实体?
                cells_all = set()
                movers_all = set()
                for e in ents:
                    n = overlap_cells(b, set(e.cells_set))
                    if n > 0:
                        cells_all.add(e.bbox)
                        movers_all |= set(e.movers)
                if not cells_all:
                    continue        # 垃圾候选上次已看够, 这次只看因果确认的
                causal_movers.append((b, a))
                print(f"  ✅因果mover: {blob_line(before, bg, b)} -> bbox={a.bbox}"
                      f"  movers={sorted(movers_all)[:4]}")
                # 锚点 = after 帧里与落点相交的 blob(fit 的 2a 逻辑用的就是这个)
                ar0, ar1, ac0, ac1 = a.bbox
                ka = a.mask_key(after, bg_a)
                found_anchor = False
                for other in ab:
                    if other.mask_key(after, bg_a) == ka:
                        continue
                    orr0, orr1, oc0, oc1 = other.bbox
                    if (min(ar1, orr1) >= max(ar0, orr0)
                            and min(ac1, oc1) >= max(ac0, oc0)):
                        found_anchor = True
                        print(f"    锚点(after帧相交): {blob_line(after, bg_a, other)}")
                if not found_anchor:
                    print(f"    ⚠️after帧无相交blob —— 落点周边: 上下左右各扩2格的色分布 "
                          f"{dict(zip(*[x.tolist() for x in np.unique(after[max(0,ar0-2):ar1+3, max(0,ac0-2):ac1+3], return_counts=True)]))}")
        print(f"  因果mover共 {len(causal_movers)} 对; mover色={sorted(set(int(c) for b,_ in causal_movers for c in b.colors))}")

    print("\n完", flush=True)


if __name__ == "__main__":
    main()
