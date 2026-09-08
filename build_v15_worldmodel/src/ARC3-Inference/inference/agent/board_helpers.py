# -*- coding: utf-8 -*-
"""预置给模型的棋盘解析库 —— 注入沙箱 runtime, 模型直接调用, 不必自己写。

依据: Retrodict 全通 25 局靠的是作者写好的 arclog.py(解析 + numpy), 模型上来
就 `steps = arclog.load()` 干活。我们原先要求模型自己攒工具, 09-08 实测 24 局
里 19 局从没定义过一个可复用函数 —— 模型不做这件事。所以工具由我们给。

只依赖 current_frame.ascii(原始数字网格在沙箱里拿不到), 全部返回朴素 Python
或 numpy 结构。
"""
HELPERS_SOURCE = r'''
def _rows_of(frame=None):
    fr = frame if frame is not None else current_frame
    a = getattr(fr, "ascii", fr)
    return a.split("\n") if isinstance(a, str) else [list(r) for r in a]


def grid(frame=None):
    """棋盘 -> numpy 字符数组(H x W), 直接用掩码/unique/argwhere。"""
    import numpy as _np
    return _np.array([list(r) for r in _rows_of(frame)], dtype="<U1")


def at(row, col, frame=None):
    """(row, col) 处的色符。"""
    return _rows_of(frame)[row][col]


def find(ch, frame=None):
    """某个色符的全部 (row, col)。"""
    return [(r, c) for r, line in enumerate(_rows_of(frame))
            for c, x in enumerate(line) if x == ch]


def counts(frame=None):
    """各色符出现次数, 多的在前。"""
    from collections import Counter
    return Counter(x for line in _rows_of(frame) for x in line).most_common()


def diff(a=None, b=None):
    """两帧之间变化的格子 [(row, col, 旧, 新)]。默认比最近一次动作的前后帧。"""
    if a is None or b is None:
        t = last_transition
        if t is None:
            return []
        a, b = t.before_frame, t.after_frame
    ra, rb = _rows_of(a), _rows_of(b)
    return [(r, c, ra[r][c], rb[r][c])
            for r in range(min(len(ra), len(rb)))
            for c in range(min(len(ra[r]), len(rb[r])))
            if ra[r][c] != rb[r][c]]


def crop(r0, r1, c0, c1, frame=None):
    """局部裁剪成多行字符串, 打印它而不是整块棋盘。"""
    return "\n".join(line[c0:c1] for line in _rows_of(frame)[r0:r1])


def objects(color=None, min_pixels=1, frame=None):
    """连通块列表(封装 segmentation): id/color/pixels/boundary/children。"""
    fr = frame if frame is not None else current_frame
    nodes = (fr.segmentation or {}).get("nodes", [])
    out = [n for n in nodes if n.get("pixels", 0) >= min_pixels]
    if color is not None:
        out = [n for n in out if n.get("color") == color]
    return sorted(out, key=lambda n: -n.get("pixels", 0))


def moved(a=None, b=None):
    """变化格子的包围盒 (r0, r1, c0, c1); 无变化返回 None。"""
    d = diff(a, b)
    if not d:
        return None
    rs = [x[0] for x in d]
    cs = [x[1] for x in d]
    return (min(rs), max(rs), min(cs), max(cs))
'''
