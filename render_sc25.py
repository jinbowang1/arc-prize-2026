"""把 sc25 的画面渲染成字符图, 对照实体坐标看懂它在干什么。

已知(diag_sc25_effect.py, 用"走两次"看效果):
    14/27 个动作有效; 实体 12 个
    A3            -> 改 行19..22 列35..38 (16 格)
    A6(x,50/55/60) -> 各改 9 格, 排成 3x3 九宫格 (行49..61 x 列24..36)
    按键效果**状态相关**, 点击效果**恒定**
两个互相矛盾的目标假设:
    修指纹后: region_match((19,22,35,42) = (19,22,23,30))   4x8  vs 4x8
    修表征后: region_match((19,22,31,42) = (50,53,11,22))   4x12 vs 4x12
到底哪块是答案区、哪块是题面, 看图说话。
"""
from __future__ import annotations

import numpy as np

from harness.env import Action, Game, action_space
from harness.percept import analyze

PAL = " .:-=+*#%@$&XO<>"      # 颜色 0..15 -> 字符


def show(g: np.ndarray, title: str, boxes: dict[str, tuple] = None) -> None:
    print(f"\n=== {title} ===", flush=True)
    print("    " + "".join(str(c % 10) for c in range(64)), flush=True)
    for r in range(64):
        row = "".join(PAL[v % 16] for v in g[r])
        marks = ""
        if boxes:
            for nm, (r0, r1, c0, c1) in boxes.items():
                if r0 <= r <= r1:
                    marks += f" <{nm}"
        print(f"{r:>3} {row}{marks}", flush=True)


game, obs = Game.make("sc25")
g0 = np.array(obs.grid)
sp = action_space(list(obs.actions))
sc = analyze(obs.grid)
acts = [Action.key(i) for i in sp["keys"]] + [Action.click(c, r) for (r, c) in sc.targets]
game.detect_lag(acts)

show(g0, "开局 (颜色->字符: " + " ".join(f"{i}={PAL[i]}" for i in sorted(set(g0.flatten().tolist()))) + ")",
     {"A": (19, 22, 31, 42), "B": (50, 53, 11, 22), "C": (19, 22, 23, 30), "九宫格": (49, 61, 24, 36)})

print("\n=== 三个候选区域的实际内容 ===", flush=True)
for nm, (r0, r1, c0, c1) in {"A(19-22,31-42)": (19, 22, 31, 42),
                             "B(50-53,11-22)": (50, 53, 11, 22),
                             "C(19-22,23-30)": (19, 22, 23, 30),
                             "D(19-22,35-42)": (19, 22, 35, 42)}.items():
    sub = g0[r0:r1 + 1, c0:c1 + 1]
    print(f"\n{nm}  形状{sub.shape} 颜色{sorted(set(sub.flatten().tolist()))}", flush=True)
    for row in sub:
        print("    " + "".join(PAL[v % 16] for v in row), flush=True)

# 按键改了什么(走两次看真效果)
print("\n=== 按键的真效果(走两次) ===", flush=True)
for k in sp["keys"]:
    a = Action.key(k)
    o = game.effect(a)
    if o.dead:
        print(f"A{k}: 致死", flush=True); continue
    g = np.array(o.grid)
    d = np.argwhere(g != g0)
    if not len(d):
        print(f"A{k}: 无变化", flush=True); continue
    rows = sorted({int(r) for r, _ in d}); cols = sorted({int(c) for _, c in d})
    print(f"A{k}: 改 {len(d)} 格, 行{rows[0]}..{rows[-1]} 列{cols[0]}..{cols[-1]}", flush=True)
    for r in rows:
        seg = "".join(PAL[g[r, c] % 16] if (r, c) in {(int(x), int(y)) for x, y in d}
                      else "·" for c in range(cols[0], cols[-1] + 1))
        old = "".join(PAL[g0[r, c] % 16] for c in range(cols[0], cols[-1] + 1))
        print(f"    行{r:>3}: {old}  ->  {seg}", flush=True)
