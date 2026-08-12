"""关卡交接:和上一关做 diff, 决定哪些知识继承、哪些作废、新增了什么要探。

**为什么必须有这一层:**

三局都是**逐关升级机制** —— ls20 每关加一种改变形状/颜色的机关, ft09 的
标记语义逐关加(中心色→三态环→开关块→lights-out), tr87 到 L5 连"可编辑的
是哪一区"都换了。所以两个极端都是错的:

  - **全不复用**(现在的 harness): 每关从零 probe, 上一关辛苦学到的槽结构、
    动作模型、目标假设全部扔掉重来。慢, 而且把免费的监督信号浪费了。
  - **全复用**: 上一关的常数直接拿来用。r11l 实测代价 —— `color_count(色 0
    的格数达到 74)` 这种从上一关抄来的常数, 在新关卡上通过了全部检验,
    白烧九分钟。

正解是**复用但不全用**: 每一条继承来的知识都要在新关卡的开局帧上过一次
体检, 过了才用, 没过要说明为什么没过。而**没过的那些, 恰恰指向这一关新加了
什么** —— 差异本身就是这一关的题目。

**diff 出来的"新元素"是 ReAct 阶段的探测目标。** 人类玩家就是这么干的:
用户玩 cd82 时的原话是"一上来就看到了(小面板), 但不知道有什么用, 当第二
还是第三关, 发现有颜色变化的时候, 多种颜色才意识到可以换油漆桶的颜色" ——
**先登记新元素, 再逐步理解它的功能**, 这两件事是分开的。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .env import Action, Obs
from .percept import Scene, analyze, background, by_figure, canonical


@dataclass
class LevelBrief:
    """一关的交接单。它同时是 ReAct 阶段的任务清单。"""

    level: int
    first: bool = False

    new_colors: list[int] = field(default_factory=list)
    gone_colors: list[int] = field(default_factory=list)
    new_shapes: int = 0              # 上一关没见过的形状等价类数量
    bg_change: tuple[int, int] | None = None
    size_change: str = ""
    n_targets: tuple[int, int] | None = None     # (上一关, 这一关) 点击候选数

    inherited: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    to_probe: list[str] = field(default_factory=list)   # 新元素, 要定向探

    def text(self) -> str:
        if self.first:
            return f"[交接] L{self.level+1} 是第一关, 无可继承知识, 全部要探"
        out = [f"[交接] L{self.level+1} 与上一关的差异:"]
        if self.new_colors:
            out.append(f"  🆕新增颜色 {self.new_colors} —— 新机制多半挂在这上面")
        if self.gone_colors:
            out.append(f"  消失颜色 {self.gone_colors}")
        if self.new_shapes:
            out.append(f"  🆕新增形状等价类 {self.new_shapes} 种")
        if self.bg_change:
            out.append(f"  ⚠️背景色变了 {self.bg_change[0]} -> {self.bg_change[1]}"
                       f" —— 一切按背景切分的感知都要重来")
        if self.n_targets:
            out.append(f"  点击候选 {self.n_targets[0]} -> {self.n_targets[1]}")
        if self.inherited:
            out.append("  ✅继承: " + "; ".join(self.inherited))
        if self.dropped:
            out.append("  ❌作废: " + "; ".join(self.dropped))
        if self.to_probe:
            out.append("  🔍待探(ReAct 的题目): " + "; ".join(self.to_probe))
        if len(out) == 1:
            out.append("  与上一关无可见差异 —— 上一关的知识大概率整套可用")
        return "\n".join(out)


@dataclass
class Knowledge:
    """跨关携带的知识。每一条都带着"它是在哪一关学到的"。

    ⚠️别把这个当成一个越攒越多的口袋。**每关都要过体检**, 过不了的当场丢,
    留着比丢掉更危险 —— 一条从上一关抄来的常数, 在新关卡上照样能通过梯度
    检验(它在旧关卡上确实下降过), 于是排在候选第一位, 白烧掉搜索预算。
    """

    goals: list = field(default_factory=list)          # (GoalHypothesis, note)
    slots: object | None = None                        # 上一关的 SlotModel
    models: dict = field(default_factory=dict)         # 上一关的 ActionModel 表
    shapes: set = field(default_factory=set)           # 见过的形状等价类
    colors: set = field(default_factory=set)
    learned_at: int = -1


def shape_keys(grid: np.ndarray) -> set:
    """这一帧里出现的形状等价类。

    用 canonical form 而不是精确像素 —— tr87 定案: 对称符号同类不同朝向,
    精确键永远去重不下来, 判定其实定义在形状等价类上。
    """
    bg = background(grid)
    out = set()
    for b in by_figure(grid, bg):
        if 1 < b.size < 400:
            out.add(canonical(b.patch(grid) != bg))
    return out


def brief(level: int, obs: Obs, scene: Scene, know: Knowledge | None,
          prev_scene: Scene | None = None) -> LevelBrief:
    """做一份交接单。纯观察, 不动真机也不动克隆体。"""
    g = np.array(obs.grid)
    b = LevelBrief(level=level, first=know is None or know.learned_at < 0)
    colors = set(int(x) for x in np.unique(g))
    shapes = shape_keys(g)

    if b.first:
        return b

    b.new_colors = sorted(colors - know.colors)
    b.gone_colors = sorted(know.colors - colors)
    b.new_shapes = len(shapes - know.shapes)
    if prev_scene is not None:
        if prev_scene.bg != scene.bg:
            b.bg_change = (prev_scene.bg, scene.bg)
        b.n_targets = (len(prev_scene.targets), len(scene.targets))

    # 新元素 = ReAct 的题目。**功能未知也要登记** —— "探测不到功能就当它
    # 不存在"是 cd82 L4 卡几小时的根因。
    for c in b.new_colors:
        n = int((g == c).sum())
        pos = np.argwhere(g == c)
        bb = (int(pos[:, 0].min()), int(pos[:, 0].max()),
              int(pos[:, 1].min()), int(pos[:, 1].max()))
        b.to_probe.append(f"色{c}({n}格, bbox={bb})")
    if b.new_shapes:
        b.to_probe.append(f"{b.new_shapes} 种没见过的形状")
    return b


def vet_goals(know: Knowledge, obs: Obs, as_heuristic, fork) -> tuple[list, list[str]]:
    """继承来的目标假设过体检: 在这一关的开局帧上还说得通吗?

    三种当场淘汰:
      - 定位不到目标对象(距离返回哨兵值)
      - 开局距离已经是 0 —— 那这个量和本关通关无关, 推它等于没推
        (cd82 L3 原样翻版: 起始 h = 0, 最佳优先当场退化成广度优先)
      - 绝对型假设且这一关的画面结构变了 —— 参数是上一关的常数
    """
    keep, drop = [], []
    for h, note in know.goals:
        try:
            d0 = as_heuristic(h)(fork(), obs)
        except Exception as e:                       # noqa: BLE001
            drop.append(f"{h.describe()} — 算不出距离({type(e).__name__})")
            continue
        if d0 >= 999:
            drop.append(f"{h.describe()} — 本关定位不到目标对象")
        elif d0 == 0:
            drop.append(f"{h.describe()} — 开局 h 已是 0, 与本关通关无关")
        else:
            keep.append((h, note, d0))
    return keep, drop


def vet_slots(know: Knowledge, live: list[Action]) -> tuple[object | None, str]:
    """继承来的槽结构过体检: 这一关的有效动作还落在那些槽里吗?

    判据是覆盖率 —— 上一关的槽里, 有多大比例的动作在这一关仍然有效。
    低于一半就别继承了, 重新分槽比在错的分组上剪枝安全。
    ⚠️分槽错了会**漏解**(某些序列被剪掉), 漏解可以错解不行, 但漏得太多就
    等于没搜。
    """
    sm = know.slots
    if sm is None:
        return None, "上一关没分过槽"
    known = {repr(a) for s in sm.slots for a in s} | {repr(a) for a in sm.loners}
    now = {repr(a) for a in live}
    if not known:
        return None, "上一关的槽是空的"
    hit = len(known & now) / len(known)
    if hit < 0.5:
        return None, f"上一关的槽只有 {hit:.0%} 的动作在这一关还有效, 重新分槽"
    return sm, f"槽结构继承({hit:.0%} 的动作仍有效)"
