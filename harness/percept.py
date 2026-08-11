"""对象化感知层:把 64x64 的颜色索引网格变成"对象"。

它解决两件事:

1. **点击空间塌缩**。click 游戏的原始动作空间是 4096 个坐标, 直接搜必死。
   ft09 定案: 有效点击目标是连通块(4096 -> ~20)。对象化是 click 游戏的
   入场费, 不是死刑。
2. **形状等价类**。tr87 定案(用户人类对照实验推翻 U=T): 判定往往定义在
   形状等价类上, 朝向不是自由度。用 canonical form 当键, 不能用精确像素
   ——精确匹配会把渲染噪声当规则本体。

已知缺口(ft09 L5 实测, 两次逃过探测)在这里正面处理: **杂色块**。
同色连通只能切出单色对象, 一个由多种颜色拼成的部件会被切碎。所以这里
同时提供两种分割, 下游拿两套都试:
    - `by_color`: 同色 4-连通(经典)
    - `by_figure`: 非背景 4-连通(忽略色差, 抓杂色部件)
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

import numpy as np

_OPS = (
    lambda m: m,
    lambda m: np.rot90(m, 1),
    lambda m: np.rot90(m, 2),
    lambda m: np.rot90(m, 3),
    lambda m: np.fliplr(m),
    lambda m: np.flipud(m),
    lambda m: m.T,
    lambda m: np.rot90(m.T, 2),
)


def canonical(m: np.ndarray) -> tuple:
    """形状等价类的规范形: 八种二面体变换里字典序最小的那个。

    tr87 教训——符号匹配必须用这个当键。同类不同朝向的符号用精确像素
    永远去重不下来, 谓词枚举就会 0 命中。
    """
    m = np.asarray(m)
    return min(tuple(map(tuple, op(m).tolist())) for op in _OPS)


@dataclass(frozen=True)
class Blob:
    """一个对象。"""

    cells: tuple[tuple[int, int], ...]
    colors: tuple[int, ...]          # 该块用到的颜色(升序去重)
    bbox: tuple[int, int, int, int]  # r0, r1, c0, c1 闭区间

    @property
    def size(self) -> int:
        return len(self.cells)

    @property
    def height(self) -> int:
        return self.bbox[1] - self.bbox[0] + 1

    @property
    def width(self) -> int:
        return self.bbox[3] - self.bbox[2] + 1

    @property
    def center(self) -> tuple[int, int]:
        return ((self.bbox[0] + self.bbox[1]) // 2, (self.bbox[2] + self.bbox[3]) // 2)

    @property
    def is_multicolor(self) -> bool:
        return len(self.colors) > 1

    def patch(self, grid: np.ndarray) -> np.ndarray:
        r0, r1, c0, c1 = self.bbox
        return grid[r0:r1 + 1, c0:c1 + 1]

    def shape_key(self, grid: np.ndarray) -> tuple:
        """规范形状键(带颜色)。"""
        return canonical(self.patch(grid))

    def mask_key(self, grid: np.ndarray, bg: int) -> tuple:
        """规范形状键(只看占位, 不看颜色)。"""
        return canonical((self.patch(grid) != bg).astype(int))


def background(grid: np.ndarray) -> int:
    """出现最多的颜色即背景。"""
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _flood(grid: np.ndarray, seed_ok, visited: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    h, w = grid.shape
    q = deque([(r, c)])
    visited[r, c] = True
    out = []
    while q:
        y, x = q.popleft()
        out.append((y, x))
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and seed_ok(y, x, ny, nx):
                visited[ny, nx] = True
                q.append((ny, nx))
    return out


def _collect(grid: np.ndarray, cells: list[tuple[int, int]]) -> Blob:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    cols = sorted({int(grid[r, c]) for r, c in cells})
    return Blob(cells=tuple(sorted(cells)), colors=tuple(cols),
                bbox=(min(rs), max(rs), min(cs), max(cs)))


def by_color(grid: np.ndarray, bg: int | None = None, min_size: int = 1) -> list[Blob]:
    """同色 4-连通分割。经典做法, 切不出杂色部件。"""
    g = np.asarray(grid)
    bg = background(g) if bg is None else bg
    visited = np.zeros(g.shape, bool)
    out = []
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if visited[r, c] or g[r, c] == bg:
                continue
            cells = _flood(g, lambda y, x, ny, nx: g[ny, nx] == g[y, x], visited, r, c)
            if len(cells) >= min_size:
                out.append(_collect(g, cells))
    return out


def by_figure(grid: np.ndarray, bg: int | None = None, min_size: int = 1) -> list[Blob]:
    """非背景 4-连通分割, 忽略色差。

    这是补 ft09 L5 那个缺口用的: 开关块是杂色格纹, 同色连通把它切成一地
    碎片, 两次逃过探测。"探测不到"不等于"不存在"。
    """
    g = np.asarray(grid)
    bg = background(g) if bg is None else bg
    visited = np.zeros(g.shape, bool)
    out = []
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if visited[r, c] or g[r, c] == bg:
                continue
            cells = _flood(g, lambda y, x, ny, nx: g[ny, nx] != bg, visited, r, c)
            if len(cells) >= min_size:
                out.append(_collect(g, cells))
    return out


def click_targets(grid: np.ndarray, bg: int | None = None) -> list[tuple[int, int]]:
    """把 4096 个点击坐标塌缩成"每个对象一个代表点"。

    两种分割都取, 去重。宁可多几个候选, 也不要漏掉杂色部件。
    """
    g = np.asarray(grid)
    bg = background(g) if bg is None else bg
    pts: list[tuple[int, int]] = []
    for blob in by_color(g, bg) + by_figure(g, bg):
        # 代表点取块内实际存在的格子, 而不是 bbox 中心(凹形块的中心可能在块外)
        cy, cx = blob.center
        if (cy, cx) in set(blob.cells):
            p = (cy, cx)
        else:
            p = blob.cells[len(blob.cells) // 2]
        if p not in pts:
            pts.append(p)
    return pts


def grid_period(grid: np.ndarray, axis: int, max_p: int = 32) -> int | None:
    """检测网格沿某轴的重复周期(用于发现单元格尺寸/棋盘结构)。"""
    g = np.asarray(grid)
    n = g.shape[axis]
    for p in range(1, min(max_p, n // 2) + 1):
        a = g[:n - p] if axis == 0 else g[:, :n - p]
        b = g[p:] if axis == 0 else g[:, p:]
        if np.array_equal(a, b):
            return p
    return None


@dataclass
class Scene:
    """一帧的对象化描述。它是 ask_human() 报告的第二部分。"""

    bg: int
    color_blobs: list[Blob]
    figure_blobs: list[Blob]
    targets: list[tuple[int, int]]
    period_r: int | None
    period_c: int | None

    def text(self, grid: np.ndarray, top: int = 8) -> str:
        g = np.asarray(grid)
        out = [f"[percept] 背景色={self.bg} 同色块={len(self.color_blobs)} "
               f"非背景块={len(self.figure_blobs)} 点击候选={len(self.targets)}"]
        if self.period_r or self.period_c:
            out.append(f"  网格周期: 行={self.period_r} 列={self.period_c}")
        multi = [b for b in self.figure_blobs if b.is_multicolor]
        if multi:
            out.append(f"  杂色块 {len(multi)} 个(同色分割会切碎它们): "
                       + ", ".join(f"{b.bbox}尺寸{b.height}x{b.width}色{b.colors}" for b in multi[:4]))
        shapes = Counter(b.mask_key(g, self.bg) for b in self.figure_blobs)
        out.append(f"  形状等价类 {len(shapes)} 种, 最常见的出现 {shapes.most_common(1)[0][1] if shapes else 0} 次")
        big = sorted(self.figure_blobs, key=lambda b: -b.size)[:top]
        for b in big:
            out.append(f"    块 bbox={b.bbox} {b.height}x{b.width} 格数={b.size} 色={b.colors}")
        return "\n".join(out)


def analyze(grid: list[list[int]] | np.ndarray) -> Scene:
    g = np.asarray(grid)
    bg = background(g)
    return Scene(
        bg=bg,
        color_blobs=by_color(g, bg),
        figure_blobs=by_figure(g, bg),
        targets=click_targets(g, bg),
        period_r=grid_period(g, 0),
        period_c=grid_period(g, 1),
    )
