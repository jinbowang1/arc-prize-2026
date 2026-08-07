"""ls20 逐步追踪器: 重放序列, 每步打印 关卡/钥匙位/是否移动/能量条/面板形状/计数器。"""
import argparse
import numpy as np
import arc_agi
from arcengine import GameAction

A = {1: GameAction.ACTION1, 2: GameAction.ACTION2, 3: GameAction.ACTION3, 4: GameAction.ACTION4}


def key_pos(g):
    pos = np.argwhere(g == 12)
    pos = pos[(pos[:, 0] < 60) & (pos[:, 1] > 12)]  # 排除左下面板/底栏
    return (int(pos[:, 0].min()), int(pos[:, 1].min())) if len(pos) else None


def panel_bits(g):
    """左下面板 9 形状 -> 3x3 位串 (每 2x2 块采样)。面板 6x6 大约 rows 55-60, cols 3-8。"""
    out = []
    for br in range(3):
        row = ""
        for bc in range(3):
            row += "X" if (g[55 + br * 2: 57 + br * 2, 3 + bc * 2: 5 + bc * 2] == 9).any() else "."
        out.append(row)
    return "/".join(out)


def bar_len(g):
    return int((g[61] == 11).sum())


def counter(g):
    return "".join(str(v) for v in g[61, 55:64])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", required=True)
    ap.add_argument("--tail", type=int, default=999, help="只打印最后 N 步")
    args = ap.parse_args()

    arc = arc_agi.Arcade()
    env = arc.make("ls20")
    f = env.reset()
    g = np.array(f.frame[-1])
    print(f"RESET lv=0 key={key_pos(g)} bar={bar_len(g)} panel={panel_bits(g)} cnt={counter(g)}")
    seq = [int(x) for x in args.actions.split(",") if x.strip()]
    prev_pos = key_pos(g)
    logs = []
    for i, a in enumerate(seq):
        f = env.step(A[a])
        g = np.array(f.frame[-1])
        pos = key_pos(g)
        moved = "" if pos != prev_pos else "  <BLOCKED"
        logs.append(f"s{i+1:3d} a{a} lv={f.levels_completed} key={pos} bar={bar_len(g):2d} "
                    f"panel={panel_bits(g)} cnt={counter(g)} {f.state.name}{moved}")
        prev_pos = pos
    for line in logs[-args.tail:]:
        print(line)


if __name__ == "__main__":
    main()
