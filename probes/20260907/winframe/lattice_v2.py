"""格点检测 v2 — 用 Retrodict 轨迹里的真实帧验证(有标准答案可对)。

判据换成: 把棋盘按候选周期 p 切成 p 列一组, 看"同一相位的列彼此相似度"。
真正的格点周期会让 board[:, i::p] 各列高度一致。
"""
import re, sys, glob, os
import numpy as np

def load_boards(log_path, max_boards=6):
    """从 Retrodict log.txt 解析 (step, board) 列表。"""
    out, rows, in_board = [], [], False
    step = None
    for line in open(log_path, encoding='utf-8', errors='replace'):
        line = line.rstrip('\n')
        m = re.match(r'\[STEP (\d+)\]', line)
        if m:
            step = int(m.group(1)); continue
        if line.startswith('[BOARD]'):
            in_board, rows = True, []; continue
        if in_board:
            vals = line.split()
            if len(vals) == 64 and all(v.isdigit() for v in vals):
                rows.append([int(v) for v in vals])
                if len(rows) == 64:
                    out.append((step, np.array(rows, dtype=np.int16)))
                    in_board = False
                    if len(out) >= max_boards: return out
            elif rows:
                in_board = False
    return out

def detect_period(board, axis, max_p=20):
    """找"分隔线"(整列/整行近乎单色)并看它们的间距 -> 格点周期。

    返回 (period, 置信度)。置信度 = 间距一致的分隔线占比。
    """
    b = board if axis == 1 else board.T
    n = b.shape[1]
    # 一列是分隔线: 该列取值高度集中(最常见值占比 >= 0.95)
    sep = []
    for i in range(n):
        col = b[:, i]
        vals, cnt = np.unique(col, return_counts=True)
        if cnt.max() / len(col) >= 0.95:
            sep.append(i)
    if len(sep) < 3:
        return None, 0.0
    # 连续的分隔列聚成一组(分隔条常是 2 列宽), 取每组的起始位置
    groups = [sep[0]]
    for a, b_ in zip(sep, sep[1:]):
        if b_ - a > 1:
            groups.append(b_)
    if len(groups) < 3:
        return None, 0.0
    real = np.diff(groups)
    if len(real) == 0:
        return None, 0.0
    vals, cnt = np.unique(real, return_counts=True)
    p = int(vals[cnt.argmax()])
    conf = float(cnt.max() / len(real))
    return p, conf


if __name__ == "__main__":
    games = sys.argv[1:] or ["ft09"]
    for g in games:
        logs = glob.glob(os.path.expanduser(
            f"~/Desktop/project/arc-agi-3/reference/retrodict-runs/release-runs/{g}/*/workspace/log.txt"))
        if not logs:
            print(f"[{g}] 没有轨迹"); continue
        boards = load_boards(logs[0])
        if not boards:
            print(f"[{g}] 没解析到棋盘"); continue
        step, b = boards[0]
        pc, sc = detect_period(b, 1)
        pr, sr = detect_period(b, 0)
        print(f"[{g}] step={step} 颜色数={len(np.unique(b))}")
        print(f"   横向: 分隔线间距={pc} (占比 {sc:.2f})")
        print(f"   纵向: 分隔线间距={pr} (占比 {sr:.2f})")


def detect_by_objects(board, min_cells=4):
    """第二种检测: 从连通域对象的尺寸/位置反推格点(适用于无分隔线的 sprite 网格)。

    取各对象包围盒的宽、高，以及各对象左上角坐标两两之差，求最大公约数。
    """
    from math import gcd
    from functools import reduce
    H, W = board.shape
    seen = np.zeros_like(board, dtype=bool)
    boxes = []
    for r in range(H):
        for c in range(W):
            if seen[r, c]:
                continue
            color = board[r, c]
            stack, cells = [(r, c)], []
            seen[r, c] = True
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                    ny, nx = y+dy, x+dx
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and board[ny, nx] == color:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if len(cells) >= min_cells:
                ys = [p[0] for p in cells]; xs = [p[1] for p in cells]
                boxes.append((min(ys), min(xs), max(ys)-min(ys)+1, max(xs)-min(xs)+1))
    if len(boxes) < 3:
        return None, None, 0
    # 背景那一大块会污染 gcd, 去掉面积最大的
    boxes = sorted(boxes, key=lambda b: -(b[2]*b[3]))[1:]
    if len(boxes) < 3:
        return None, None, 0
    tops = sorted(set(b[0] for b in boxes))
    lefts = sorted(set(b[1] for b in boxes))
    def g(vals):
        # gcd 对噪声极敏感(一个离格对象就退化成 1), 改用相邻间距的众数
        d = [abs(b - a) for a, b in zip(vals, vals[1:]) if b != a]
        if not d:
            return 0
        vs, cs = np.unique(np.array(d), return_counts=True)
        return int(vs[cs.argmax()])
    return g(tops), g(lefts), len(boxes)


if os.environ.get("LATTICE_OBJ") == "1":
    for g_ in (sys.argv[1:] or ["ft09"]):
        logs = glob.glob(os.path.expanduser(
            f"~/Desktop/project/arc-agi-3/reference/retrodict-runs/release-runs/{g_}/*/workspace/log.txt"))
        if not logs: continue
        bs = load_boards(logs[0])
        if not bs: continue
        pr, pc, n = detect_by_objects(bs[0][1])
        print(f"[{g_}] 对象法: 纵向格点={pr} 横向格点={pc} (基于 {n} 个对象)")
