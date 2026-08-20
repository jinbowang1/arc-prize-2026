"""sc25 L4-L6 通用攻关 —— ⚠️白盒方案(源码见 attack_sc25_l3.py 头注)。

法术全谱(源码破译):
  fibcey  竖线(0,1)(1,1)(2,1)   = 喷火, 方向=最后按的方向键; 打中 tagsmh 清
                                   {tagsmh+全部dosorb}, 打中 seofsw-tagsmh 清 seofsw 系
  sieesc  菱形(0,1)(1,0)(1,2)(2,1) = 变身: scale 1↔2 (变大需空间, 不够则失败闪烁)
  tevyeq  角形(0,0)(0,1)(1,1)    = 传送到当前激活的 tevyeq-tagsmh 标记(逐次轮转,
                                   激活标记有 acyylh-tevyeq-inkpfx 指示物=像素可见)

搜索: 宏动作 Dijkstra(按真实步数最短)。宏 = {A1..A4, 施火, 变身, 传送}。
状态指纹 = (全帧 bytes, pending) —— 朝向(玩家旋转像素)/传送轮转(指示物位置)/
尺寸全都在像素里, 指纹完备。画布施法后自动清空, 宏之间画布恒空, 中间态不会
误触发别的图案(三种图案的绘制顺序都验过无前缀撞车)。
过关判定 = level 翻转, 或补 3 步结算后翻转(触出口有 2 帧走出动画)。
"""
from __future__ import annotations

import heapq
import json
import re
import time

import numpy as np

from harness.env import Action, Game

CLICK = re.compile(r"A6\((\d+),(\d+)\)")

FIRE = [Action.click(30, 50), Action.click(30, 55), Action.click(30, 60)]
MORPH = [Action.click(30, 50), Action.click(25, 55), Action.click(35, 55), Action.click(30, 60)]
TELEPORT = [Action.click(25, 50), Action.click(30, 50), Action.click(30, 55)]

MACROS = [
    ("A1", [Action.key(1)]), ("A2", [Action.key(2)]),
    ("A3", [Action.key(3)]), ("A4", [Action.key(4)]),
    ("火", FIRE), ("变", MORPH), ("传", TELEPORT),
]


def to_action(tok):
    m = CLICK.match(tok)
    if m:
        return Action.click(int(m.group(1)), int(m.group(2)))
    return Action.key(int(tok[1]))


def win_after_settle(node: Game, base_level: int, extra: int = 3) -> int:
    c = node.fork()
    for k in range(extra):
        o = c.act(Action.key(1))
        if o.level > base_level:
            return k + 1
    return -1


def solve_level(base: Game, lv: int, spells: set[str],
                max_expand: int = 40000, max_seconds: float = 1200.0):
    """从 base(停在 lv 开局)出发, 宏 Dijkstra 找过关序列。返回 Action 列表或 None。"""
    macros = [(n, s) for n, s in MACROS
              if n in ("A1", "A2", "A3", "A4")
              or (n == "火" and "fibcey" in spells)
              or (n == "变" and "sieesc_chwjgc" in spells)
              or (n == "传" and "tevyeq" in spells)]
    t0 = time.time()
    cnt = 0
    heap = [(0, cnt, base.fork(), [])]
    seen = set()
    expanded = 0
    while heap and expanded < max_expand and time.time() - t0 < max_seconds:
        cost, _, node, path = heapq.heappop(heap)
        expanded += 1
        for name, seq in macros:
            n = node.fork()
            o = None
            dead = False
            for a in seq:
                o = n.act(a)
                if o.dead:
                    dead = True
                    break
            if dead:
                continue
            npath = path + list(seq)
            if o.level > lv:
                print(f"  L{lv+1} ✅ 找到! {len(npath)} 步, 展开 {expanded} 状态, {time.time()-t0:.0f}s")
                return npath
            # 指纹遮掉步数条(action-ui 在 x62-63, 每步变一格): 不遮的话
            # 同一局面经不同步数到达永远去重不了, 图搜索退化成分层树。
            g = np.array(o.grid)
            fp = (g[:, :60].tobytes(), o.pending)
            if fp in seen:
                continue
            seen.add(fp)
            k = win_after_settle(n, lv)
            if k > 0:
                npath = npath + [Action.key(1)] * k
                print(f"  L{lv+1} ✅ 找到(补{k}步结算)! {len(npath)} 步, 展开 {expanded} 状态, {time.time()-t0:.0f}s")
                return npath
            cnt += 1
            heapq.heappush(heap, (cost + len(seq), cnt, n, npath))
        if expanded % 200 == 0:
            print(f"  L{lv+1} 展开 {expanded}, 队列 {len(heap)}, 已见 {len(seen)}, {time.time()-t0:.0f}s", flush=True)
    print(f"  L{lv+1} ❌ 未找到 (展开 {expanded}, {time.time()-t0:.0f}s)")
    return None


LEVEL_SPELLS = {
    3: {"fibcey", "sieesc_chwjgc"},
    4: {"fibcey", "tevyeq", "sieesc_chwjgc"},
    5: {"fibcey", "tevyeq", "sieesc_chwjgc"},
}


def main():
    sol = json.load(open("sc25_solutions.json"))
    game, obs = Game.make("sc25")
    for t in sol["seq"]:
        obs = game.act(to_action(t))
    print(f"在案解重放完成: level={obs.level} state={obs.state}", flush=True)
    assert obs.level == 3, "L1-L3 重放失败"

    while obs.level < obs.win_levels:
        lv = obs.level
        print(f"\n===== 攻 L{lv+1} (人类基准 {sol['baseline'][lv]}) =====", flush=True)
        path = solve_level(game, lv, LEVEL_SPELLS[lv])
        if path is None:
            break
        # 新克隆整条复核
        v = game.fork()
        o = None
        for a in path:
            o = v.act(a)
        ok = o.level > lv or win_after_settle(v, lv) > 0
        print(f"  整条复核: {'✅' if ok else '❌'}", flush=True)
        if not ok:
            break
        for a in path:
            obs = game.act(a)
        if obs.level <= lv:
            print("  ⚠️真机没翻关, 停", flush=True)
            break
        sol["seq"] = sol["seq"] + [str(a) for a in path]
        sol["per_level_steps"] = sol["per_level_steps"] + [len(path)]
        json.dump(sol, open("sc25_solutions.json", "w"), ensure_ascii=False)
        print(f"  ✅ L{lv+1} 已过({len(path)}步), 解已写入", flush=True)

    print(f"\n最终: level={obs.level}/{obs.win_levels} state={obs.state}", flush=True)


if __name__ == "__main__":
    main()
