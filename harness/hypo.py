"""假设引擎:提出"目标是什么", 并给出**有梯度**的启发式。

为什么是目标而不是转移: Tycho 的消融给了实证 —— transition 模拟精度
16.2% 却拿 88.49 分, 88.1% 精度只拿 83.07 分。**目标识别比动力学精度
更决定分数**。三局的亲身经历也一致: ls20 卡在读锁、tr87 卡在判定定义在
形状等价类上、ft09 卡在蓝图语义, 没有一关是卡在"走不动"。

🚨**启发式必须有梯度**(ls20 L6 定案): "不对就罚 6 分"这类常数罚项会让
"差一次操作"和"差三次操作"同分, 最佳优先立刻退化成广度搜索。所以每个
假设都必须能给出一个随着接近目标而单调下降的实数。

假设从**已通关关卡的通关瞬间**自动拟合 —— 这是免费的监督信号, 每通一关
就多一个样本。人工只在自动拟合全部落空时介入, 且只能给"族名 + 参数"。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .percept import Blob, analyze, background, by_figure, canonical


@dataclass
class Transition:
    """一个通关瞬间的样本。"""

    before: np.ndarray
    after: np.ndarray
    level: int


class GoalHypothesis:
    """一个目标假设。子类必须同时给出判定和**有梯度的**距离。"""

    name = "base"

    def is_goal(self, grid: np.ndarray) -> bool:
        raise NotImplementedError

    def distance(self, grid: np.ndarray) -> float:
        """到目标的估计剩余代价。必须随接近目标单调下降, 不能用常数罚项。"""
        raise NotImplementedError

    def describe(self) -> str:
        return self.name


@dataclass
class ObjectReach(GoalHypothesis):
    """假设: 把某个形状的对象移动到某个位置。

    距离 = 该对象当前位置到目标位置的曼哈顿距离。天然有梯度。
    ls20 全部七关、以及任何"走到某处"的游戏都落在这一族。
    """

    shape_key: tuple
    target: tuple[int, int]
    name: str = "object_reach"

    def _find(self, grid: np.ndarray) -> tuple[int, int] | None:
        bg = background(grid)
        for b in by_figure(grid, bg):
            if b.mask_key(grid, bg) == self.shape_key:
                return b.center
        return None

    def is_goal(self, grid: np.ndarray) -> bool:
        return self._find(grid) == self.target

    def distance(self, grid: np.ndarray) -> float:
        p = self._find(grid)
        if p is None:
            return 999.0
        return abs(p[0] - self.target[0]) + abs(p[1] - self.target[1])

    def describe(self) -> str:
        return f"object_reach(把某 {len(self.shape_key)}行形状 移到 {self.target})"


@dataclass
class ObjectToObject(GoalHypothesis):
    """假设: 把形状 A 的对象移到形状 B 的对象所在处。

    🚨这一族是 cd82 盲测暴露的清单缺口(人工补族 #1)。原先只有 ObjectReach,
    它把目标位置记成从上一关学来的**绝对坐标**, 换一关就废: cd82 L3 起始
    h 直接 = 0, 最佳优先退化回广度优先。

    **目标假设必须是关系型的, 不能是绝对坐标** —— 目标位置要由当前帧里的
    另一个对象决定。ls20 的"走进锁房"、ft09 的"填色匹配蓝图"、tr87 的
    "答案区匹配字典"本质全是关系型。这条应该硬编码进比赛 harness 当先验。
    """

    mover: tuple
    anchor: tuple
    name: str = "object_to_object"

    def _locate(self, grid: np.ndarray) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
        bg = background(grid)
        m = a = None
        for b in by_figure(grid, bg):
            k = b.mask_key(grid, bg)
            if k == self.mover and m is None:
                m = b.center
            elif k == self.anchor and a is None:
                a = b.center
        return m, a

    def is_goal(self, grid: np.ndarray) -> bool:
        m, a = self._locate(grid)
        return m is not None and m == a

    def distance(self, grid: np.ndarray) -> float:
        m, a = self._locate(grid)
        if m is None or a is None:
            return 999.0
        return abs(m[0] - a[0]) + abs(m[1] - a[1])

    def describe(self) -> str:
        return (f"object_to_object(把 {len(self.mover)}行的块 "
                f"移到 {len(self.anchor)}行的块所在处)")


@dataclass
class RegionMatch(GoalHypothesis):
    """假设: 区域 A 的内容要和区域 B 一致(蓝图/查表/填色类)。

    距离 = 两区域不一致的格数。逐格递减, 有梯度。
    ft09 的蓝图填色、tr87 的答案区匹配都落在这一族。
    """

    a: tuple[int, int, int, int]
    b: tuple[int, int, int, int]
    name: str = "region_match"

    def _pair(self, grid: np.ndarray):
        r0, r1, c0, c1 = self.a
        p0, p1, q0, q1 = self.b
        x = grid[r0:r1 + 1, c0:c1 + 1]
        y = grid[p0:p1 + 1, q0:q1 + 1]
        if x.shape != y.shape:
            return None, None
        return x, y

    def is_goal(self, grid: np.ndarray) -> bool:
        x, y = self._pair(grid)
        return x is not None and bool((x == y).all())

    def distance(self, grid: np.ndarray) -> float:
        x, y = self._pair(grid)
        if x is None:
            return 999.0
        return float((x != y).sum())

    def describe(self) -> str:
        return f"region_match({self.a} 要等于 {self.b})"


@dataclass
class ColorCount(GoalHypothesis):
    """假设: 某颜色的格数要达到某个目标值(清零/填满/计数)。

    距离 = |当前计数 - 目标计数|。有梯度。
    """

    color: int
    target: int
    name: str = "color_count"

    def _n(self, grid: np.ndarray) -> int:
        return int((grid == self.color).sum())

    def is_goal(self, grid: np.ndarray) -> bool:
        return self._n(grid) == self.target

    def distance(self, grid: np.ndarray) -> float:
        return float(abs(self._n(grid) - self.target))

    def describe(self) -> str:
        return f"color_count(色 {self.color} 的格数达到 {self.target})"


@dataclass
class ColorAppear(GoalHypothesis):
    """假设: 某个颜色出现(完成标记/亮灯)。

    ⚠️这一族**没有梯度**——出现前距离恒为 1。它只适合当判定, 不适合当
    启发式。保留它是因为完成标记是最常见的通关信号, 但用它做搜索会退化
    成广度搜索, 所以 distance 故意返回一个大常数以示警告。
    """

    color: int
    name: str = "color_appear"

    def is_goal(self, grid: np.ndarray) -> bool:
        return bool((grid == self.color).any())

    def distance(self, grid: np.ndarray) -> float:
        return 0.0 if self.is_goal(grid) else 50.0

    def describe(self) -> str:
        return f"color_appear(出现色 {self.color}) ⚠️无梯度, 仅作判定"


def fit(samples: list[Transition]) -> list[GoalHypothesis]:
    """从通关瞬间自动拟合候选目标假设, 按"梯度质量"排序。

    判据: 假设必须在 after 成立、在 before 不成立。在所有样本上都满足的
    假设才留下 —— 单样本拟合出来的东西基本都是巧合。
    """
    if not samples:
        return []
    cands: list[GoalHypothesis] = []

    # 1) ColorAppear: after 有而 before 没有的颜色
    for s in samples[:1]:
        new_colors = set(np.unique(s.after).tolist()) - set(np.unique(s.before).tolist())
        cands += [ColorAppear(int(c)) for c in sorted(new_colors)]

    # 2) 移动型目标: 前后同形状但位置变了的块
    s = samples[0]
    bg = background(s.before)
    bg_a = background(s.after)
    before_blobs = by_figure(s.before, bg)
    after_blobs = by_figure(s.after, bg_a)
    for b in before_blobs:
        kb = b.mask_key(s.before, bg)
        for a in after_blobs:
            if a.mask_key(s.after, bg_a) != kb or a.center == b.center:
                continue
            # 2a) 关系型(优先): 它移到了谁身上? 找 after 里与它同心的另一个块
            for other in after_blobs:
                ko = other.mask_key(s.after, bg_a)
                if ko != kb and other.center == a.center:
                    cands.append(ObjectToObject(kb, ko))
            # 2b) 绝对坐标(退化版, 跨关不可用, 仅当关系型拟合不出时兜底)
            cands.append(ObjectReach(kb, a.center))
            break

    # 3) ColorCount: 某色格数在 after 达到某个值且 before 不等
    for c in np.unique(s.after):
        na, nb = int((s.after == c).sum()), int((s.before == c).sum())
        if na != nb:
            cands.append(ColorCount(int(c), na))

    # 用全部样本过滤: 必须条条样本 after 成立 before 不成立
    kept = []
    for h in cands:
        if all(h.is_goal(t.after) and not h.is_goal(t.before) for t in samples):
            kept.append(h)

    # 排序: 关系型优先(唯一能跨关泛化的), 其次有梯度的, 无梯度的垫底
    def rank(h: GoalHypothesis) -> tuple:
        return (not isinstance(h, ObjectToObject), isinstance(h, ColorAppear), h.name)

    kept.sort(key=rank)
    return kept


FAMILIES = {
    "object_to_object": ObjectToObject,
    "object_reach": ObjectReach,
    "region_match": RegionMatch,
    "color_count": ColorCount,
    "color_appear": ColorAppear,
}


def ask_human_report(probe_text: str, percept_text: str, search_text: str,
                     samples: list[Transition], fitted: list[GoalHypothesis]) -> str:
    """卡住时打给人看的定长报告。人只能回一行"族名 + 参数"。

    分层覆盖诊断(tr87 L6 教训): 0 命中时要逐族说明为什么没命中, 而不是
    只说一句"没找到" —— 那次正是分层诊断顺手暴露了回文布局。
    """
    lines = [
        "=" * 60,
        "【harness 卡住, 请求人工给一条谓词族假设】",
        "=" * 60,
        probe_text,
        percept_text,
        search_text,
        f"\n[hypo] 通关样本 {len(samples)} 个, 自动拟合出 {len(fitted)} 条候选:",
    ]
    if fitted:
        for h in fitted[:8]:
            lines.append(f"    - {h.describe()}")
    else:
        lines.append("    (无) 逐族说明:")
        for name in FAMILIES:
            lines.append(f"    - {name}: 未从通关样本中拟合出在所有样本上都成立的实例")
    lines += [
        "\n可选族: " + ", ".join(FAMILIES),
        "请回一行: <族名> <参数>   (回不上来才用逃生舱, 逃生舱会单独计数)",
        "=" * 60,
    ]
    return "\n".join(lines)
