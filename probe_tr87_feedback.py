"""tr87 的决定性实验: 填错了有没有反馈?

这一条决定我的搜索流水线在这类游戏上还有没有抓手:
  - 有逐位反馈  -> 每位独立试 7 种, 共 5x7=35 次试探就能定位答案, 搜索可行
  - 全或无      -> 必须一次填对 5 位, 盲搜 7^5=16807 种组合 x 每种约 20 步 = 33 万步
                   而人类基准是 54 步 -> 搜索彻底失效, 只能靠归纳

做法: 枚举第1位的全部 7 种符号, 每种都检查画面上除该位以外有没有任何变化。
"""
import copy
import numpy as np
import arc_agi
from arcengine import GameAction, ActionInput

ACTS = {i: getattr(GameAction, f"ACTION{i}") for i in range(1, 8)}
POS = [15, 22, 29, 36, 43]                 # 5 个符号位的起始列


def raw(g, a): return g.perform_action(ActionInput(id=ACTS[a], data={}), raw=True)


def clone(g):
    clean = getattr(g, "_clean_levels", None)
    g._clean_levels = None; g2 = copy.deepcopy(g)
    g._clean_levels = clean; g2._clean_levels = clean
    return g2


def cell(g, r0, c0):
    return tuple(map(tuple, (g[r0:r0 + 5, c0:c0 + 5] == 5).astype(int)))


def show(m):
    return "/".join("".join("X" if v else "." for v in row) for row in m)


arc = arc_agi.Arcade()
env = arc.make("tr87")
f = env.reset()
game = env._game
g0 = np.array(f.frame[-1])

print("枚举第1位的 7 种符号, 看画面其它地方有无任何反馈:\n")
ch = clone(game)
prev = g0
for i in range(8):
    fr = raw(ch, 1)
    if not fr.frame:
        print("  无帧"); break
    g = np.array(fr.frame[-1])
    d = np.argwhere(g != prev)
    # 排除"当前编辑位"本身(行51-57 列15-19)的变化, 看别处动没动
    other = [(int(r), int(c)) for r, c in d
             if not (51 <= r <= 57 and 15 <= c <= 19)]
    rows = sorted({r for r, _ in other})
    print(f"  第{i+1}次 符号={show(cell(g, 52, 15))}")
    print(f"       本位外变动 {len(other)} 像素" + (f", 涉及行 {rows}" if other else "  <- 无任何反馈")
          + f" | 关卡={fr.levels_completed}")
    prev = g

print("\n" + "=" * 60)
print("再验一次: 把 5 个位置各随便改几下, 看有没有'对了几位'之类的指示")
ch = clone(game)
prev = np.array(f.frame[-1])
plan = [(1, 3), (4, 1), (1, 2), (4, 1), (1, 5), (4, 1), (1, 1), (4, 1), (1, 4)]
for a, times in plan:
    for _ in range(times):
        fr = raw(ch, a)
        if not fr.frame:
            break
g = np.array(fr.frame[-1])
d = np.argwhere(g != np.array(f.frame[-1]))
rows = sorted({int(r) for r, _ in d})
print(f"  乱填一通后: 变动像素 {len(d)}, 涉及行 {rows}")
print(f"  关卡={fr.levels_completed} 状态={fr.state.name}")
print("  若变动只出现在 答案区(51-57) 和 光标(48-49,59-60), 就说明没有任何对错提示。")
