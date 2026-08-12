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


@dataclass
class Entity:
    """画面上一个"会动的东西"。

    ⚠️**登记实体与理解功能是两件事。** 人类对照实验(2026-08-11, 用户亲玩
    cd82)的原话:"一上来就看到了(小面板), 但不知道有什么用, 第二三关发现
    有颜色变化的时候才意识到可以换油漆桶的颜色。"

    所以功能未知的实体**也必须留在状态表征里**。我在 cd82 L4 上正是因为
    "探测不到用途就当它不存在", 把第二个面板整个丢了 —— 状态表征少一个
    自由度, 后面搜多久都没用。
    """

    bbox: tuple[int, int, int, int]
    movers: list[str]                 # 哪些动作会改动它
    cells: int                        # 典型变化格数
    role: str = "unknown"             # 功能待定;未定也要留着
    cells_set: tuple = ()             # 具体是哪些格(可变掩码要用)

    def line(self) -> str:
        return (f"实体 bbox={self.bbox} {self.cells}格 "
                f"受 {len(self.movers)} 个动作影响 {self.movers[:5]} 角色={self.role}")


def discover(peek, base_grid: np.ndarray, actions: list,
             min_cells: int = 2) -> tuple[list[Entity], np.ndarray]:
    """同时交出实体表和**可变格掩码**(至少被某个动作改过的格子)。

    可变格掩码本来就是实体发现的中间产物, 以前算完就扔了。它是免费的因果
    信息, 而且是目标识别的关键: **动作能改的是答案区, 改不了却有内容的是
    题面**(见 hypo.propose_prompt_answer)。
    """
    ents = discover_entities(peek, base_grid, actions, min_cells)
    m = np.zeros(base_grid.shape, dtype=bool)
    for e in ents:
        for (r, c) in e.cells_set:
            m[r, c] = True
    return ents, m


def mutable_over_states(peeks: list, grids: list[np.ndarray],
                        actions: list) -> np.ndarray:
    """在**多个状态**上求可变格的并集。

    🚨只在开局那一个状态上采会低估可变区, 而且低估得很难看: cd82 的答案区
    是 10×10, 只从开局采出来是 **5×10** —— 因为 A5 那支笔在开局的面板位置
    只盖得到上半区, 下半区要等面板移动过才够得着。答案区少一半, 题面配对
    就再也对不上尺寸, 目标识别整条链就断在这里。

    这是"采样只在一个状态上做"这个错的第三次变形(前两次: 动作候选在起始态
    算一次、抽象模型只在开局采)。**凡是"这个游戏里 X 能不能被改变"这类
    问题, 都必须在多个状态上问。**

    `peeks[i](action) -> grid` 与 `grids[i]` 一一对应。
    """
    m = np.zeros(grids[0].shape, dtype=bool)
    for peek, g in zip(peeks, grids):
        for a in actions:
            g1 = np.asarray(peek(a))
            if g1.shape == g.shape:
                m |= (g1 != g)
    return m


def discover_entities(peek, base_grid: np.ndarray, actions: list,
                      min_cells: int = 2) -> list[Entity]:
    """对每个动作做帧 diff, 把**互不重叠的变化区**各自登记成独立实体。

    这是 cd82 L4 那道坎的根治办法。当时画面上有两个面板(大 12×7、小 4×3),
    我只跟踪了最显眼的大的;小面板归 A3/A4 管, 它一移动, 点它盖出的位置就
    跟着变。而我把动作候选在起始态算了一次就当全局事实, 于是那支能补上
    缺口的笔在整个搜索里根本不存在 —— 抽象层和真机 beam 双双卡死, 加宽
    beam 加到 800 也没用, 因为答案不在候选里。

    发现它靠的是人工全屏渲染。这个函数就是把那一眼自动化:
    不问"这个动作有什么用", 只问"画面上有几处互不相干的地方在动"。

    `peek(action) -> grid` 由调用方提供(通常是 game.peek 的包装)。

    🚨**判据是"总是一起变的格子属于同一个实体", 不是"变化区有重叠就合并"。**
    第一版按重叠合并, 在 r11l(纯 click, 每次点击重绘一大片)上实测直接崩:
    所有动作的变化区两两重叠, 一路并到底, 报出**一个 396 格、bbox 覆盖
    整屏的"实体"** —— 等于什么都没发现, 而且没发现得很像发现了。

    正解是给每个格子算一个**签名 = 会改动它的动作集合**, 签名相同的格子
    归为一个实体。大面板归 A5、小面板归 A3/A4, 签名不同就分得开, 哪怕
    它们的变化区在别处有交叠。
    """
    sig: dict[tuple[int, int], set[str]] = {}
    for a in actions:
        g1 = np.asarray(peek(a))
        if g1.shape != base_grid.shape:
            continue
        changed = np.argwhere(g1 != base_grid)
        if len(changed) < min_cells:
            continue
        for r, c in changed:
            sig.setdefault((int(r), int(c)), set()).add(repr(a))

    groups: dict[frozenset, list[tuple[int, int]]] = {}
    for cell, movers in sig.items():
        groups.setdefault(frozenset(movers), []).append(cell)

    out = []
    for movers, cells in groups.items():
        if len(cells) < min_cells:
            continue
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        out.append(Entity(bbox=(min(rs), max(rs), min(cs), max(cs)),
                          movers=sorted(movers), cells=len(cells),
                          cells_set=tuple(cells)))
    out.sort(key=lambda e: -e.cells)
    return out


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
