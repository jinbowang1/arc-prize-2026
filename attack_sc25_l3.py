"""sc25 L3 攻关 —— ⚠️白盒方案(读了 environment_files/sc25/*/sc25.py 源码)。

源码破译(混淆名→语义):
  · 游戏 = 画符咒施法。3×3 画布(点击格子切换亮灭)匹配到本关必修咒语的图案
    才施法; 匹配到非必修的图案**毫无反应**(这解释了当年 512 构型穷尽无效)。
  · L3 必修 fibcey = 中列竖线 [[F,T,F],[F,T,F],[F,T,F]] → 喷火法术。
  · 喷火方向 = jdmucabyqar = **最后一次按的方向键**(A1上/A2下/A3左/A4右) ——
    当年推断的"不可见内部状态"就是它。
  · 火球打中 tagsmh(55,22) → tagsmh 与所有 dosorb(27,34, 挡路石)一起消失。
  · **过关条件 = 玩家(pluyoo)走到出口 exydhv(22,37)**, 与画布无关 ——
    当年"画布×方块位置穷尽无解"的根因: 赢的维度根本不在里面。
  · 施法 8 帧 + 火球飞行/收缩若干帧, 期间任何动作只推进动画不结算移动。

方案(全在克隆体上验证, 真机只走验证过的序列):
  ① A4 面朝右(玩家(35,22)与 tagsmh(55,22) 同行)
  ② 点 (30,50)(30,55)(30,60) 画中列竖线 → 施法自动触发
  ③ 用 A3 喂动画(动画期间被吞; 动画结束后 A3=向左走=朝出口方向, 失败安全)
  ④ BFS 走到出口(触到 exydhv 后还有 2 帧走出动画, 补 3 步结算)
"""
from __future__ import annotations

import itertools
import json
import re

import numpy as np

from harness.env import Action, Game

CLICK = re.compile(r"A6\((\d+),(\d+)\)")


def to_action(tok):
    if isinstance(tok, str):
        m = CLICK.match(tok)
        if m:
            return Action.click(int(m.group(1)), int(m.group(2)))
        return Action.key(int(tok[1]))
    return Action.key(int(tok))


def win_after_settle(node: Game, base_level: int, extra: int = 3) -> int:
    """触出口后有 2 帧走出动画, 补几步无害动作看 level 翻没翻。返回所需补步数, -1=没赢。"""
    c = node.fork()
    for k in range(extra):
        o = c.act(Action.key(1))
        if o.level > base_level:
            return k + 1
    return -1


def main():
    sol = json.load(open("sc25_solutions.json"))
    game, obs = Game.make("sc25")
    for t in sol["seq"]:
        obs = game.act(to_action(t))
    print(f"L1+L2 重放完成: level={obs.level} state={obs.state}", flush=True)
    assert obs.level == 2, "前两关重放失败"

    l3_grid = np.array(game.fork().act(Action.key(1)).grid)  # 只为读帧, 用克隆
    base = game  # 真机停在 L3 开局, 下面全部用 fork 试

    # ①② 面右 + 画竖线
    prelude = [Action.key(4),
               Action.click(30, 50), Action.click(30, 55), Action.click(30, 60)]

    # ③ 实测(diag): 施法+火球动画在第三下点击的**同一次调用**里一口气跑完
    #    (layers=16), 不吃后续动作 —— 不需要喂步。直接确认挡路石已开。
    c = base.fork()
    g0 = np.array(c._grid())
    for a in prelude:
        c.act(a)
    if np.array_equal(np.array(c._grid())[33:43, 26:36], g0[33:43, 26:36]):
        print("❌ 施法后挡路石区域没变, 方案有误, 停")
        return
    fed = 0
    print("✅ 施法完成, 挡路石已清", flush=True)

    # ④ 在这个克隆上 BFS 走到出口
    print("BFS 走出口...", flush=True)
    keys = [Action.key(i) for i in (1, 2, 3, 4)]
    seen = set()
    frontier = [(c, [])]
    win_seq = None
    for depth in range(1, 15):
        nxt = []
        for node, path in frontier:
            for a in keys:
                n = node.fork()
                o = n.act(a)
                if o.dead:
                    continue
                if o.level > 2:
                    win_seq = path + [a]
                    break
                fp = (np.array(o.grid).tobytes(), o.pending)
                if fp in seen:
                    continue
                seen.add(fp)
                k = win_after_settle(n, 2)
                if k > 0:
                    win_seq = path + [a] + [Action.key(1)] * k
                    break
                nxt.append((n, path + [a]))
            if win_seq:
                break
        if win_seq:
            break
        frontier = nxt
        print(f"  深度 {depth}: 前沿 {len(frontier)}", flush=True)
    if not win_seq:
        print("❌ BFS 14 层没找到出口, 停")
        return
    print(f"✅ 克隆体通关! 走位 {len(win_seq)} 步", flush=True)

    full = prelude + win_seq
    # 整条在新克隆上复核一遍
    v = base.fork()
    o = None
    for a in full:
        o = v.act(a)
    ok = o.level > 2 or win_after_settle(v, 2) > 0
    print(f"整条复核: {'✅' if ok else '❌'} ({len(full)} 步, 人类基准 {sol['baseline'][2]})", flush=True)
    if not ok:
        return

    # 真机执行
    for a in full:
        obs = base.act(a)
    print(f"真机: level={obs.level} state={obs.state} 本关步数={len(full)}", flush=True)
    if obs.level > 2:
        sol["seq"] = sol["seq"] + [str(a) for a in full]
        sol["per_level_steps"] = sol["per_level_steps"] + [len(full)]
        json.dump(sol, open("sc25_solutions.json", "w"), ensure_ascii=False)
        print("✅ L3 已过, 解已写入 sc25_solutions.json", flush=True)
    else:
        print("⚠️真机没翻关(可能差结算步), 手动查", flush=True)


if __name__ == "__main__":
    main()
