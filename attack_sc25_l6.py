"""sc25 L6(最终关)攻关 —— 白盒混合法(源码破译见 attack_sc25_l3.py)。

L6 链条: 出口(32,8)正下被 seofsw-dosorb(33,14) 堵; 其靶 seofsw-tagsmh(37,37)
在底廊但火线被 dosorb(42,37) 挡; dosorb 的靶 tagsmh(17,33) 在只有 scale1
传送(smzaik标记(13,41))能进的左下小室。传送轮转: scale2 第一次→(53,37)右下,
第二次→(29,33)中央口袋(通中廊→出口)。

计划: 缩身+传送→小室朝上火(清dosorb) → 变大+传送→右下朝左火(清seofsw-dosorb)
      → 再传送→中央 → 走位登顶。每段用实验搜索(火位枚举/走位BFS), 不硬编码坐标。
"""
from __future__ import annotations

import json
import re

import numpy as np

from harness.env import Action, Game

CLICK = re.compile(r"A6\((\d+),(\d+)\)")
FIRE = [Action.click(30, 50), Action.click(30, 55), Action.click(30, 60)]
MORPH = [Action.click(30, 50), Action.click(25, 55), Action.click(35, 55), Action.click(30, 60)]
TELEPORT = [Action.click(25, 50), Action.click(30, 50), Action.click(30, 55)]
KEYS = [Action.key(i) for i in (1, 2, 3, 4)]

TAG_A = (33, 38, 16, 22)     # tagsmh(17,33) 环
TAG_B = (37, 42, 37, 42)     # seofsw-tagsmh(37,37) 环


def to_action(tok):
    m = CLICK.match(tok)
    if m:
        return Action.click(int(m.group(1)), int(m.group(2)))
    return Action.key(int(tok[1]))


def win_after_settle(node, base_level, extra=3):
    c = node.fork()
    for k in range(extra):
        o = c.act(Action.key(1))
        if o.level > base_level:
            return k + 1
    return -1


def walk_states(start: Game, max_depth: int = 20):
    out = [([], start)]
    seen = {np.array(start._grid())[:, :60].tobytes()}
    frontier = [([], start)]
    for _ in range(max_depth):
        nxt = []
        for path, node in frontier:
            for a in KEYS:
                n = node.fork()
                o = n.act(a)
                if o.dead:
                    continue
                fp = np.array(o.grid)[:, :60].tobytes()
                if fp in seen:
                    continue
                seen.add(fp)
                nxt.append((path + [a], n))
                out.append((path + [a], n))
        if not nxt:
            break
        frontier = nxt
    return out


def fire_plans(node: Game, box, cap=8, max_depth=20):
    """(走位+FIRE后的节点, 完整路径) 列表, 按路径长升序。"""
    y0, y1, x0, x1 = box
    ref = np.array(node._grid())[y0:y1, x0:x1].copy()
    out = []
    for path, nd in walk_states(node, max_depth):
        c = nd.fork()
        dead = False
        for a in FIRE:
            o = c.act(a)
            if o.dead:
                dead = True
                break
        if dead:
            continue
        if not np.array_equal(np.array(c._grid())[y0:y1, x0:x1], ref):
            out.append((path + list(FIRE), c))
    out.sort(key=lambda x: len(x[0]))
    return out[:cap]


def apply(node: Game, seq) -> tuple[Game, bool]:
    c = node.fork()
    for a in seq:
        o = c.act(a)
        if o.dead:
            return c, False
    return c, True


def main():
    sol = json.load(open("sc25_solutions.json"))
    game, obs = Game.make("sc25")
    for t in sol["seq"]:
        obs = game.act(to_action(t))
    print(f"重放完成 level={obs.level}", flush=True)
    assert obs.level == 5
    lv = 5

    # P1 缩身+传送 → smzaik 小室
    c1, ok = apply(game, MORPH + TELEPORT)
    if not ok:
        print("❌ P1 死")
        return
    # P2 小室里找能清 tagsmh 的火位
    fa = fire_plans(c1, TAG_A, max_depth=12)
    print(f"P2 tagsmh 火位 {len(fa)} 个", flush=True)
    for pa, ca in fa:
        # P3 变大+传送 → 右下 (变大需空间, 逐走位点试)
        for w3, n3 in sorted(walk_states(ca, max_depth=8), key=lambda x: len(x[0]))[:20]:
            c3, ok = apply(n3, MORPH + TELEPORT)
            if not ok:
                continue
            # P4 右下找能清 seofsw-tagsmh 的火位
            fb = fire_plans(c3, TAG_B, cap=4, max_depth=14)
            if not fb:
                continue
            print(f"P4 seofsw 火位 {len(fb)} 个", flush=True)
            for pb, cb in fb:
                # P5 再传送 → 中央口袋
                c5, ok = apply(cb, TELEPORT)
                if not ok:
                    continue
                # P6 走位登顶
                seen = set()
                frontier = [(c5, [])]
                tail = None
                for _ in range(16):
                    nxt = []
                    for nd, path in frontier:
                        for a in KEYS:
                            n = nd.fork()
                            o = n.act(a)
                            if o.dead:
                                continue
                            if o.level > lv:
                                tail = path + [a]
                                break
                            fp = (np.array(o.grid)[:, :60].tobytes(), o.pending)
                            if fp in seen:
                                continue
                            seen.add(fp)
                            k = win_after_settle(n, lv)
                            if k > 0:
                                tail = path + [a] + [Action.key(1)] * k
                                break
                            nxt.append((n, path + [a]))
                        if tail:
                            break
                    if tail:
                        break
                    frontier = nxt
                    if not frontier:
                        break
                if tail is None:
                    continue
                full = (list(MORPH) + list(TELEPORT) + pa + w3
                        + list(MORPH) + list(TELEPORT) + pb + list(TELEPORT) + tail)
                v2 = game.fork()
                o = None
                for a in full:
                    o = v2.act(a)
                ok2 = o.level > lv or (o.state != "WIN" and win_after_settle(v2, lv) > 0) or o.state == "WIN"
                print(f"候选 {len(full)} 步 复核 {'✅' if ok2 else '❌'}", flush=True)
                if not ok2:
                    continue
                for a in full:
                    obs = game.act(a)
                if obs.level > lv:
                    sol["seq"] += [str(a) for a in full]
                    sol["per_level_steps"] += [len(full)]
                    json.dump(sol, open("sc25_solutions.json", "w"), ensure_ascii=False)
                    print(f"🏆 L6 已过({len(full)}步 vs 人类{sol['baseline'][lv]}), sc25 全通!", flush=True)
                    print(f"最终 state={obs.state} level={obs.level}/{obs.win_levels}", flush=True)
                    return
    print("❌ 本方案未通, 需要重推")


if __name__ == "__main__":
    main()
