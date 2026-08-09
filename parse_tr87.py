"""tr87 通用感知: 从帧中提取 字典对(左语言->右语言串) / 题面串 / 答案区位。

布局规律(L1/L2 实测): 背景=2(上半)/3(下半); 字典在行带 4-10/13-19/22-28,
每带两对, 框顶行是连续非背景 run; 题面=行40-46 大框; 答案区=行51-57 大框; 符号=5x5, 笔画色5。
"""
import numpy as np

def runs_in_row(row, bg):
    """返回 [(c0, c1, color)] 连续非背景 run(闭区间)"""
    out, c = [], 0
    n = len(row)
    while c < n:
        if row[c] != bg:
            c0, col = c, row[c]
            while c < n and row[c] == col:
                c += 1
            out.append((c0, c - 1, int(col)))
        else:
            c += 1
    return out

def glyphs_in_box(g, r0, c0, c1):
    """框内部(r0+1..r0+5 行, c0+1..c1-1 列)按 7 列一位切出 5x5 掩码列表"""
    out = []
    c = c0 + 1
    while c + 4 <= c1 - 1:
        m = tuple(tuple(1 if g[r0 + 1 + i][c + j] == 5 else 0 for j in range(5)) for i in range(5))
        out.append(m)
        c += 7
    return out

def nonempty(ms):
    return [m for m in ms if any(any(r) for r in m)]

def show(m):
    return "/".join("".join("X" if v else "." for v in row) for row in m)

def parse(g):
    g = np.asarray(g)
    dict_pairs = []
    for band in (4, 13, 22):
        rs = runs_in_row(g[band], 2)
        # 依次两两配对: (框0,框1) (框2,框3)
        for i in range(0, len(rs) - 1, 2):
            a0, a1, _ = rs[i]
            b0, b1, _ = rs[i + 1]
            src = glyphs_in_box(g, band, a0, a1)
            dst = glyphs_in_box(g, band, b0, b1)
            dict_pairs.append((tuple(nonempty(src)), tuple(nonempty(dst))))
    prob_run = runs_in_row(g[40], 3)
    p0, p1, _ = prob_run[0]
    problem = glyphs_in_box(g, 40, p0, p1)
    ans_run = runs_in_row(g[51], 3)
    q0, q1, _ = ans_run[0]
    answer = glyphs_in_box(g, 51, q0, q1)
    return dict_pairs, nonempty(problem), answer, (q0, q1)

if __name__ == "__main__":
    import arc_agi
    arc = arc_agi.Arcade()
    env = arc.make("tr87")
    f = env.reset()
    # 重放 L1 解到达 L2
    import json
    from arcengine import GameAction
    l1 = json.load(open("tr87_l1.json"))
    for a in l1["l1_seq"]:
        f = env.step(getattr(GameAction, f"ACTION{a}"))
    print(f"L1 重放后 levels={f.levels_completed}")
    g = np.array(f.frame[-1])
    pairs, prob, ans, span = parse(g)
    print("=== 字典 ===")
    for s, d in pairs:
        print("  src:", " ".join(show(m) for m in s))
        print("   ->:", " ".join(show(m) for m in d))
    print("=== 题面 ===")
    for m in prob:
        print("  ", show(m))
    print(f"=== 答案区 {span} {len(ans)} 位 ===")
    for m in ans:
        print("  ", show(m))
    # 完备性检查
    lut = {s: d for s, d in pairs if len(s) == 1}
    lut1 = {s[0]: d for s, d in lut.items()}
    missing = [m for m in prob if m not in lut1]
    print(f"完备性: 题面 {len(prob)} 符号, 字典可查 {len(prob) - len(missing)}, 缺 {len(missing)}")
    if not missing:
        target = [x for m in prob for x in lut1[m]]
        print(f"目标答案串({len(target)}位):")
        for m in target:
            print("  ", show(m))
