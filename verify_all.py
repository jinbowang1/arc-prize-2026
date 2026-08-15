"""四局在案满分解的整条重放核验。

只回答一个问题: **今天这四条解还打不打得通**。
每局都是全新环境从头重放, 不复用任何进程内状态 —— 搜索进程内自报通关不算数,
只有整条重放到 state=WIN 才算(ls20 L6 的边界重复陷阱就是这么抓出来的)。
"""
from __future__ import annotations

import json
import re
import time

from harness.env import Action, Game

CLICK = re.compile(r"A6\((\d+),(\d+)\)")


def to_action(tok) -> Action:
    """三种在案格式统一成 Action。

    - int          : ls20/tr87 的按键 id
    - "A3"/"A6(x,y)": cd82 的按键与点击
    - [x, y]       : ft09 的裸点击坐标
    """
    if isinstance(tok, int):
        return Action.key(tok)
    if isinstance(tok, (list, tuple)):
        return Action.click(int(tok[0]), int(tok[1]))
    m = CLICK.match(tok)
    if m:
        return Action.click(int(m.group(1)), int(m.group(2)))
    return Action.key(int(tok.replace("A", "")))


def load(path: str) -> list:
    d = json.load(open(path))
    if "seqs" in d:                      # 逐关分段存的, 拼成一条
        return [a for leg in d["seqs"] for a in leg]
    return d["seq"]


CASES = [
    ("ls20", "solutions.json", 7, 776),
    ("tr87", "tr87_solutions_canon.json", 6, 414),
    ("ft09", "ft09_solutions.json", 6, 208),
    ("cd82", "cd82_solutions.json", 6, 171),
]

print(f"{'game':6} {'步数':>5} {'人类':>5} {'关卡':>7} {'状态':>12} {'耗时':>7}")
print("-" * 52)
rows = []
for game_id, path, want_levels, human in CASES:
    seq = [to_action(t) for t in load(path)]
    t0 = time.time()
    g, obs = Game.make(game_id)
    for a in seq:
        obs = g.act(a)
        if obs.dead:
            break
    dt = time.time() - t0
    ok = obs.state == "WIN" and obs.level >= want_levels
    rows.append((game_id, len(seq), human, obs.level, want_levels, obs.state, ok))
    print(f"{game_id:6} {len(seq):>5} {human:>5} {obs.level:>3}/{want_levels:<3} "
          f"{obs.state:>12} {dt:>6.1f}s {'✅' if ok else '❌'}")

print("-" * 52)
n_ok = sum(r[-1] for r in rows)
print(f"整条重放通过 {n_ok}/{len(rows)} 局")
