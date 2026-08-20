"""sc25 L5 攻关 —— 白盒混合法(源码破译见 attack_sc25_l3.py; 结构见本文件底注)。

L5 结构: 出口(50,10)右上; 右竖廊(x52-55)里摞着 dosorb(51,16)+seofsw-dosorb(51,20);
tagsmh(15,11)左上, seofsw-tagsmh(3,43)左下室; 传送标记 tevyeq-tagsmh(51,35)在右廊底;
玩家口袋与右廊不连通 → 必须: 两次喷火清两块挡路石 + 传送进右廊 + 走到出口。

方法: ①走位 BFS 枚举可达状态(scale2 与 morph 后 scale1 各一轮)
      ②每个状态 peek 一次喷火, 看哪个靶子被清 → 得到"火A方案/火B方案"
      ③组合 火→火→传送→走位BFS, 两种火序都试, 克隆整条复核后真机执行
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

# 两个靶子的观察窗(y0,y1,x0,x1), 区域内容变了=该靶被清
TAG_A = (11, 16, 14, 20)     # tagsmh(15,11) 6环
TAG_B = (43, 48, 2, 12)      # seofsw-tagsmh(3,43)


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


def region(g, box):
    y0, y1, x0, x1 = box
    return g[y0:y1, x0:x1]


def walk_states(start: Game, max_depth: int = 22):
    """走位 BFS, 产出 (path, game) 全部可达状态(含起点)。"""
    out = [([], start)]
    seen = {(np.array(start._grid())[:, :60].tobytes(),)}
    frontier = [([], start)]
    for _ in range(max_depth):
        nxt = []
        for path, node in frontier:
            for a in KEYS:
                n = node.fork()
                o = n.act(a)
                if o.dead:
                    continue
                fp = (np.array(o.grid)[:, :60].tobytes(),)
                if fp in seen:
                    continue
                seen.add(fp)
                nxt.append((path + [a], n))
                out.append((path + [a], n))
        if not nxt:
            break
        frontier = nxt
    return out


def fire_hits(node: Game, g_ref: np.ndarray, box) -> bool:
    c = node.fork()
    for a in FIRE:
        o = c.act(a)
        if o.dead:
            return False
    return not np.array_equal(region(np.array(c._grid()), box), region(g_ref, box))


def find_fire_plans(base: Game, box, prefix: list) -> list:
    """返回能清掉 box 靶子的 (前缀+走位+FIRE) 完整路径列表(按长度升序)。"""
    c = base.fork()
    for a in prefix:
        c.act(a)
    g_ref = np.array(c._grid())
    plans = []
    for path, node in walk_states(c):
        if fire_hits(node, g_ref, box):
            plans.append(prefix + path + list(FIRE))
    plans.sort(key=len)
    return plans


def walk_to_win(node: Game, lv: int, max_depth: int = 18):
    seen = set()
    frontier = [(node, [])]
    for _ in range(max_depth):
        nxt = []
        for nd, path in frontier:
            for a in KEYS:
                n = nd.fork()
                o = n.act(a)
                if o.dead:
                    continue
                if o.level > lv:
                    return path + [a]
                fp = (np.array(o.grid)[:, :60].tobytes(), o.pending)
                if fp in seen:
                    continue
                seen.add(fp)
                k = win_after_settle(n, lv)
                if k > 0:
                    return path + [a] + [Action.key(1)] * k
                nxt.append((n, path + [a]))
        frontier = nxt
        if not frontier:
            break
    return None


def main():
    sol = json.load(open("sc25_solutions.json"))
    game, obs = Game.make("sc25")
    for t in sol["seq"]:
        obs = game.act(to_action(t))
    print(f"重放完成 level={obs.level}", flush=True)
    assert obs.level == 4
    lv = 4

    # 口袋区无任何火线(实验已证) → 路线: 缩身+传送(scale1→smzaik标记(29,39))
    # → 下部区域打两靶 → 变大+传送(scale2→普通标记(51,35)右廊) → 走到出口
    c0 = game.fork()
    for a in MORPH + TELEPORT:
        o = c0.act(a)
    if o.dead:
        print("❌ 缩身+传送阶段死了")
        return
    print("缩身+传送完成, 找第一把火(先试B=seofsw, 再试A)", flush=True)

    def fire_plans_from(node, box, cap=8):
        g_ref = np.array(node._grid())
        out = []
        for path, nd in walk_states(node, max_depth=20):
            if fire_hits(nd, g_ref, box):
                out.append((path + list(FIRE), nd))
        out.sort(key=lambda x: len(x[0]))
        return out[:cap]

    for order in ("BA", "AB"):
        box1 = TAG_B if order[0] == "B" else TAG_A
        box2 = TAG_A if order[1] == "A" else TAG_B
        first = fire_plans_from(c0, box1)
        print(f"火序{order}: 第一靶火位 {len(first)} 个", flush=True)
        for p1, n1 in first:
            c1 = n1.fork()
            for a in FIRE:
                c1.act(a)
            second = fire_plans_from(c1, box2)
            if not second:
                continue
            print(f"  第二靶火位 {len(second)} 个", flush=True)
            for p2, _ in second[:6]:
                c2 = c1.fork()
                for a in p2:
                    c2.act(a)
                # 变大+传送 -> 右廊; 变大需空间, 在各走位点上都试
                for wpath, wnode in sorted(walk_states(c2, max_depth=10), key=lambda x: len(x[0]))[:30]:
                    c3 = wnode.fork()
                    dead = False
                    for a in MORPH + TELEPORT:
                        o = c3.act(a)
                        if o.dead:
                            dead = True
                            break
                    if dead:
                        continue
                    tail = walk_to_win(c3, lv, max_depth=14)
                    if tail is None:
                        continue
                    full = (list(MORPH) + list(TELEPORT) + p1 + list(FIRE) + p2
                            + wpath + list(MORPH) + list(TELEPORT) + tail)
                    # ⚠️p1 已含 FIRE? fire_plans_from 返回 path+FIRE, 而 n1 是 FIRE 前的节点
                    # 上面 c1 又补了 FIRE, 所以 full 里 p1 应为不带 FIRE 的走位 —— 修正:
                    full = (list(MORPH) + list(TELEPORT) + p1[:-len(FIRE)] + list(FIRE)
                            + p2 + wpath + list(MORPH) + list(TELEPORT) + tail)
                    v = game.fork()
                    o = None
                    for a in full:
                        o = v.act(a)
                    ok = o.level > lv or win_after_settle(v, lv) > 0
                    print(f"  候选 {len(full)} 步 复核 {'✅' if ok else '❌'}", flush=True)
                    if not ok:
                        continue
                    for a in full:
                        obs = game.act(a)
                    if obs.level > lv:
                        sol["seq"] += [str(a) for a in full]
                        sol["per_level_steps"] += [len(full)]
                        json.dump(sol, open("sc25_solutions.json", "w"), ensure_ascii=False)
                        print(f"✅ L5 已过({len(full)}步 vs 人类{sol['baseline'][lv]}), 解已写入", flush=True)
                        return
        print(f"火序{order} 未成", flush=True)
    print("❌ 仍未通, 需要再看结构")


if __name__ == "__main__":
    main()
