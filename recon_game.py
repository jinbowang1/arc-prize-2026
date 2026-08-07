"""通用游戏侦察器: 对任意游戏, 不预设任何常量, 自动回答四个问题。

  1. 每个动作会改变画面的哪些区域?
  2. 哪个连通块是"我控制的东西"(随动作方向做位移)?
  3. 哪些区域是 HUD(随步数单调变化, 与位置无关)?
  4. 移动的步长是多少?

刻意不 import wm.py —— 那里面全是 ls20 专用常量(玩家色 12/9、面板在 55 行、能量条在 61 行)。
这个脚本是"感知层能否自动化"的第一块试金石。

用法: uv run python recon_game.py tr87
"""
import copy, sys
from collections import Counter
import numpy as np
import arc_agi
from arcengine import GameAction, ActionInput

GID = sys.argv[1] if len(sys.argv) > 1 else "tr87"
ACTS = {i: getattr(GameAction, f"ACTION{i}") for i in range(1, 8)}


def raw(g, a, data=None):
    return g.perform_action(ActionInput(id=ACTS[a], data=data or {}), raw=True)


def clone(g):
    clean = getattr(g, "_clean_levels", None)
    g._clean_levels = None
    g2 = copy.deepcopy(g)
    g._clean_levels = clean
    g2._clean_levels = clean
    return g2


def blocks(mask):
    """4-邻接连通块 -> [(外接框, 像素数)], 按大小降序。"""
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    H, W = mask.shape
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            stack, cells = [(r, c)], []
            seen[r, c] = True
            while stack:
                y, x = stack.pop(); cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True; stack.append((ny, nx))
            ys = [y for y, _ in cells]; xs = [x for _, x in cells]
            out.append(((min(ys), min(xs), max(ys), max(xs)), len(cells)))
    return sorted(out, key=lambda t: -t[1])


arc = arc_agi.Arcade()
env = arc.make(GID)
f = env.reset()
game = env._game
g0 = np.array(f.frame[-1])
print(f"=== {GID} ===")
print(f"关卡数 {f.win_levels} | 可用动作 {f.available_actions} | 帧 {g0.shape} {g0.dtype}")
print(f"颜色分布 {dict(zip(*[x.tolist() for x in np.unique(g0, return_counts=True)]))}")

CH = ".123456789ABCDEF"
print("\n--- 开局画面 ---")
print("    " + "".join(str(c % 10) for c in range(g0.shape[1])))
for r in range(g0.shape[0]):
    print(f"{r:>3} " + "".join(CH[v] for v in g0[r]))

print("\n--- 问题1&2: 每个动作改变了什么 ---")
acts = [a for a in f.available_actions if a in ACTS and a not in (6, 7)]
per_act = {}
for a in acts:
    ch = clone(game)
    fr = raw(ch, a)
    if not fr.frame:
        print(f"  ACTION{a}: 无帧(可能立即结束)"); continue
    g1 = np.array(fr.frame[-1])
    d = (g0 != g1)
    bs = blocks(d)
    per_act[a] = (g1, bs)
    print(f"  ACTION{a}: 变动 {d.sum()} 像素, {len(bs)} 个连通块")
    for (box, n) in bs[:4]:
        print(f"        框 行{box[0]}-{box[2]} 列{box[1]}-{box[3]}  {n} 像素")

print("\n--- 问题3: 连做同一个动作 8 次, 哪些区域一直在变(HUD 候选) ---")
if acts:
    a0 = acts[0]
    ch = clone(game)
    prev = g0
    changed_count = np.zeros_like(g0, dtype=int)
    for i in range(8):
        fr = raw(ch, a0)
        if not fr.frame:
            print(f"  第{i+1}步无帧"); break
        g = np.array(fr.frame[-1])
        changed_count += (g != prev)
        prev = g
    hot = changed_count >= 6          # 8 步里变了 6 次以上 = 高频区
    for (box, n) in blocks(hot)[:6]:
        print(f"  高频变动区 行{box[0]}-{box[2]} 列{box[1]}-{box[3]}  {n} 像素"
              f"   <- HUD/计时器嫌疑")

print("\n--- 问题4: 受控对象与步长 ---")
print("  判据: 同一个连通块在不同动作下朝不同方向整体位移, 位移量即步长")
if len(per_act) >= 2:
    for a, (g1, bs) in per_act.items():
        # 找"消失处"和"出现处": 原来有、现在没有 vs 原来没有、现在有
        for (box, n) in bs[:2]:
            r0, c0, r1, c1 = box
            print(f"  ACTION{a} 最大变动块: {r1-r0+1}x{c1-c0+1} @ 行{r0} 列{c0}")
