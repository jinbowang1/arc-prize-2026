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

from .percept import Blob, analyze, background, by_color, by_figure, canonical


@dataclass
class Transition:
    """一个通关瞬间的样本。"""

    before: np.ndarray
    after: np.ndarray
    level: int
    # 该关开局的实体表(percept.discover 的输出, 带 movers=谁能动)。
    # fit 用它做因果过滤: "谁是可动件"是因果信息不是尺寸特征 ——
    # MoverToAnchor 两版尺寸猜(最小/最大)错向相反, 就是缺这个接口。
    ents: list = field(default_factory=list)


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

    def _locate(self, grid: np.ndarray):
        """找 mover 与 anchor 的 bbox。

        🚨**两种分割都取**, 且必须与 `fit` 生成候选时用的是同一套 ——
        原先这里只用 by_figure, 而它把所有非背景色一锅烩: ls20 上色 3/5/9/12
        彼此相邻连成 971 格巨块, 5x5 的钥匙整个被吞掉, 于是判定阶段**永远
        找不到 mover**, is_goal 恒假, fit 生成的 68 条候选被自己全部过滤光。
        """
        bg = background(grid)
        blobs = list(by_figure(grid, bg)) + list(by_color(grid, bg))
        m = a = None
        for b in blobs:
            k = b.mask_key(grid, bg)
            if k == self.mover and m is None:
                m = b.bbox
            elif k == self.anchor and a is None:
                a = b.bbox
        return m, a

    def is_goal(self, grid: np.ndarray) -> bool:
        """判据 = **bbox 有实质重叠**, 不是中心完全重合。

        🚨必须与 `fit` 生成候选时的判据**一致**。原先生成用"重叠"、判定用
        "同心", 生成宽判定严 —— 等于批量生产注定被自己否掉的候选。
        而 ls20 的钥匙走进锁房本来就是**重叠**(两者差一格), 要求同心就永远不成立。
        """
        m, a = self._locate(grid)
        if m is None or a is None:
            return False
        return (min(m[1], a[1]) >= max(m[0], a[0])
                and min(m[3], a[3]) >= max(m[2], a[2]))

    def distance(self, grid: np.ndarray) -> float:
        """距离 = 两 bbox 的轴向间隙之和(行 gap + 列 gap)。

        🚨原版是 `abs(m[0]-a[0]) + abs(m[1]-a[1])` —— m 是 (r0,r1,c0,c1),
        这两项是行起点差+**行终点差**, **列坐标根本没参与**。ls20 L2 实锤:
        钥匙要横向走 17 格进锁房, 横向移动零梯度, 真机最佳优先搜到 65 层
        h 卡在 5(那 5 就是两个行差)纹丝不动。ObjectToObject 修好生成链后
        仍一关没解过, 死因就在这。
        gap 版的另一个好处: distance=0 ⟺ 两轴都无间隙 ⟺ bbox 重叠 ——
        与 is_goal 的判据**同语义**(判定和启发式用两套语义的亏吃过多次)。
        """
        m, a = self._locate(grid)
        if m is None or a is None:
            return 999.0
        gap_r = max(0, max(m[0], a[0]) - min(m[1], a[1]))
        gap_c = max(0, max(m[2], a[2]) - min(m[3], a[3]))
        return float(gap_r + gap_c)

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
class ColorCountMatch(GoalHypothesis):
    """假设: 色 A 的格数要等于色 B 的格数(填满/配对/消除类)。

    🚨这一族是 r11l 盲测暴露的清单缺口。原先的 ColorCount 把目标值记成一个
    **从上一关抄来的常数**("色 0 的格数要达到 74"), 换一关立刻变成噪声,
    却还能通过梯度检验 —— 因为它在上一关上确实从 2 掉到了 0。三条这样的
    常数型假设在 r11l L2 上白烧了九分钟。

    **能跨关的假设, 参数必须由当前帧决定。** 把"达到 74"换成"等于另一个色
    的格数", 同一条假设在每一关自动重新取值。这和 ObjectToObject 把绝对
    坐标换成关系锚点是同一个道理, 也是 hypo 模块开头那句"目标假设必须是
    关系型的"的第二次应验。
    """

    a: int
    b: int
    name: str = "color_count_match"

    def _n(self, grid: np.ndarray, c: int) -> int:
        return int((grid == c).sum())

    def is_goal(self, grid: np.ndarray) -> bool:
        return self._n(grid, self.a) == self._n(grid, self.b)

    def distance(self, grid: np.ndarray) -> float:
        return float(abs(self._n(grid, self.a) - self._n(grid, self.b)))

    def describe(self) -> str:
        return f"color_count_match(色 {self.a} 的格数要等于色 {self.b} 的格数)"


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
    # 🚨**两种分割都取**。by_figure 忽略色差(ft09 的杂色开关块靠它, 同色连通
    # 会切成碎片); 但它把**所有非背景色一锅烩**, ls20 上色 3/5/9/12 彼此相邻,
    # 连成一块 **971 格、跨 42x40** 的巨块, 5x5 的钥匙被整个吞掉 ——
    # 于是 ObjectToObject(钥匙匹配锁 = ls20 的真判据)**一条都拟合不出**,
    # 只剩 ColorCount 那种"色 3 的格数达到 919"的绝对型目标被继承到下一关。
    # by_color 按颜色分别连通, 立刻能分出钥匙(色 12 在 (45,34) 10 格 /
    # 色 9 在 (47,34) 15 格)。
    # percept.click_targets 早就写着"两种分割都取, 去重 —— 宁可多几个候选,
    # 也不要漏掉杂色部件", fit 却只用了一种。
    def _blobs(g, b):
        out = list(by_figure(g, b)) + list(by_color(g, b))
        seen, uniq = set(), []
        for x in out:
            if x.bbox not in seen:
                seen.add(x.bbox)
                uniq.append(x)
        return uniq

    before_blobs = _blobs(s.before, bg)
    after_blobs = _blobs(s.after, bg_a)

    # 因果过滤: mover 候选必须与"带 movers 的实体"有格重叠。
    # ls20 七关在案解重放实测(diag_mover_causal.log): 真 mover(钥匙色12/色9)
    # 全部与实体重叠, 而 1x1 碎点等垃圾候选全部不在任何实体里 ——
    # 垃圾 mover 生成的假关系型目标会挤占 max_goals 名额。
    # 没有实体信息(ents 为空)时不过滤, 退回旧行为。
    ent_cells: set = set()
    for e in getattr(s, "ents", None) or []:
        if getattr(e, "movers", None):
            ent_cells |= set(e.cells_set)

    def _causally_movable(blob) -> bool:
        if not ent_cells:
            return True
        r0, r1, c0, c1 = blob.bbox
        return any(r0 <= r <= r1 and c0 <= c <= c1 for (r, c) in ent_cells)

    for b in before_blobs:
        if not _causally_movable(b):
            continue
        kb = b.mask_key(s.before, bg)
        for a in after_blobs:
            if a.mask_key(s.after, bg_a) != kb or a.center == b.center:
                continue
            # 2a) 关系型(优先): 它移到了谁身上?
            # 🚨判据是"**有实质重叠**", 不是"中心完全重合"。
            # ls20 实测: 钥匙走进锁房时两者只是靠近/重叠, 中心差一格,
            # 于是 ObjectToObject **一条都拟合不出**(同形状块对 4、移动过的 2、
            # 而"同心"的 0), 只剩 ColorCount 那种"色 3 的格数达到 1356"的
            # 绝对型垃圾目标被继承到下一关 —— ls20 L2 的 h 卡在 3 就是被它带偏的。
            ar0, ar1, ac0, ac1 = a.bbox
            for other in after_blobs:
                ko = other.mask_key(s.after, bg_a)
                if ko == kb:
                    continue
                orr0, orr1, oc0, oc1 = other.bbox
                # 两个 bbox 相交即算"移到了它身上"
                if min(ar1, orr1) >= max(ar0, orr0) and min(ac1, oc1) >= max(ac0, oc0):
                    cands.append(ObjectToObject(kb, ko))
            # 2b) 绝对坐标(退化版, 跨关不可用, 仅当关系型拟合不出时兜底)
            cands.append(ObjectReach(kb, a.center))
            # ⚠️不再 break: 原来处理完第一个匹配块就退出, 后面的块根本没机会 ——
            # 场上不止一个东西会动(ls20 有钥匙也有巡逻体)。

    # 3) RegionMatch: 同尺寸的块两两配对 —— "把这块弄成那块的样子"
    #    (此前定义了这一族却从没在这里枚举过, 属实现漏项)
    boxes = [b.bbox for b in after_blobs]
    for i, x in enumerate(boxes):
        for y in boxes[i + 1:]:
            if (x[1] - x[0], x[3] - x[2]) == (y[1] - y[0], y[3] - y[2]):
                cands.append(RegionMatch(x, y))
                cands.append(RegionMatch(y, x))

    # 4) ColorCount: 某色格数在 after 达到某个值且 before 不等
    #    ⚠️目标值是常数, 跨关即失效, 只是兜底。真正能跨关的是下面的配对版。
    for c in np.unique(s.after):
        na, nb = int((s.after == c).sum()), int((s.before == c).sum())
        if na != nb:
            cands.append(ColorCount(int(c), na))

    # 5) ColorCountMatch: 两个色的格数在 after 相等而在 before 不等 —— 关系型,
    #    参数由当前帧自己取值, 换关不用改。
    colors = [int(c) for c in np.unique(np.concatenate([s.before.ravel(), s.after.ravel()]))]
    for i, ca in enumerate(colors):
        for cb in colors[i + 1:]:
            cands.append(ColorCountMatch(ca, cb))

    # 用全部样本过滤: 必须条条样本 after 成立 before 不成立
    kept = []
    for h in cands:
        if all(h.is_goal(t.after) and not h.is_goal(t.before) for t in samples):
            kept.append(h)

    # 去重: 同族同参数的假设会被重复生成(实测 r11l 上一口气冒出 21 条
    # 一模一样的 object_reach)。重复项会挤占"只试前 N 条"的预算,
    # 看起来像试了很多条, 其实反复在试同一条。
    uniq, seen = [], set()
    for h in kept:
        k = h.describe()
        if k not in seen:
            seen.add(k)
            uniq.append(h)

    # 排序: 关系型优先(唯一能跨关泛化的), 其次有梯度的, 无梯度的垫底
    def rank(h: GoalHypothesis) -> tuple:
        return (not is_relational(h), isinstance(h, ColorAppear), h.name)

    uniq.sort(key=rank)
    return uniq


def is_relational(h: GoalHypothesis) -> bool:
    """这条假设的参数是不是由当前帧决定的?

    只有关系型假设能跨关用。绝对型(ObjectReach 的坐标、ColorCount 的目标值、
    RegionMatch 的 bbox)带的是上一关的常数, 到了新关卡是噪声 —— 而且是能
    通过梯度检验的噪声, 因为它在上一关上确实下降过。报告里必须标出来。
    """
    return isinstance(h, (ObjectToObject, ColorCountMatch))


@dataclass
class SubmitMatch(GoalHypothesis):
    """假设: 存在一个"提交"动作, 提交后某个结果区要匹配某个题面区。

    🚨这一族是 cd82 L3 的逃生舱产出(人工介入 #2), 但形式通用: 凡是"先调整
    再提交、提交即判定"的游戏都落在这里 —— tr87 的答案区、ft09 的填色、
    cd82 的印章盖印, 都是同一个形状。

    它的价值在于**距离可以隔着提交动作算**: 在任意中间状态上 peek 一次提交
    动作(克隆体, 不花真实步数), 拿盖印结果与题面比对, 差异格数就是距离。
    这比"猜中间状态该长什么样"直接得多, 而且天然有梯度。

    ⚠️它需要 game 句柄而不只是网格, 所以走 distance_on_node 而不是
    distance —— 这是这一族与其它族的接口差别。
    """

    submit: object                       # Action, 提交动作
    result: tuple[int, int, int, int]    # 提交后看哪块
    target: tuple[int, int, int, int]    # 要匹配的题面块
    name: str = "submit_match"

    def _cmp(self, grid: np.ndarray, after: np.ndarray) -> float:
        r0, r1, c0, c1 = self.result
        t0, t1, u0, u1 = self.target
        got = after[r0:r1 + 1, c0:c1 + 1]
        want = grid[t0:t1 + 1, u0:u1 + 1]
        if got.shape != want.shape:
            return 999.0
        # ⚠️别用 "!= 背景色" 判占位: cd82 的题面里"暗格"是色 0 而不是背景色 5,
        # 那样两边会全判成"有内容", 差异恒为 0, 启发式在起点就归零。
        # 逐格比色号, 差异格数天然有梯度。
        return float((got != want).sum())

    def distance_on_node(self, node, obs, base_level: int) -> float:
        child = node.fork()
        o = child.act(self.submit)
        if o.level > base_level:
            return 0.0
        if o.dead:
            return 999.0
        return self._cmp(np.array(obs.grid), np.array(o.grid))

    def is_goal(self, grid: np.ndarray) -> bool:
        return False          # 判定只认引擎的 levels_completed

    def distance(self, grid: np.ndarray) -> float:
        return 999.0          # 本族必须用 distance_on_node

    def describe(self) -> str:
        return f"submit_match(按 {self.submit} 后 {self.result} 要匹配 {self.target})"


def propose_prompt_answer(grid: np.ndarray, mutable: np.ndarray,
                          bg: int | None = None, max_pairs: int = 12
                          ) -> list[RegionMatch]:
    """从**因果结构**生成目标假设: 让"我能改的那块"变得和"我改不了的那块"一样。

    🚨这一族是 2026-08-12 攻目标识别时补的, 而它其实是四局共同的形状:

        cd82  答案区(10×10) 要等于 题面(10×10)
        ft09  填色区 要等于 蓝图
        tr87  答案区 要等于 字典给出的译文
        ls20  钥匙的形状+颜色 要等于 锁上显示的

    以前 RegionMatch 的实例是从**同尺寸 blob 两两配对**凑出来的 —— 那是
    像素巧合, 一关能凑出几十条, 且参数是绝对坐标, 换关即废。

    这里换成因果判据: **动作能改的区域是答案区, 动作改不了却有内容的区域
    是题面。** 这个划分每关重算, 所以它天然跨关 —— 复用的是"目标 = 答案区
    匹配题面"这条**族**, 参数由当前帧自己决定。

    因果信息是免费的: `mutable` 就是感知层做实体发现时顺手得到的
    "至少被某个动作改过的格子", 不额外花 fork。

    ⚠️只提尺寸完全相同的配对。尺寸不同要做缩放/平移匹配, 那是另一回事,
    没证据之前不做 —— 宁可漏, 不可错。
    """
    if bg is None:
        bg = background(grid)
    h, w = grid.shape

    # 答案区候选: 可变格的连通块, 取其 bbox。
    # 🚨要跑**两种连通粒度**: 紧(4 邻域)和松(距离<=GAP)。
    # 画布可能是**分离格子组成的网格**, 格与格之间有分隔行/列 —— 紧连通会把它
    # 切成 N 块, 每块单独成候选, 整块画布反而提不出来。
    # sc25 L3 实测: 九宫格(13x13, 9 个 3x3 格、间隔 2)被切成 9 个候选, 每个只
    # 对应 **1 个**提交动作; 而正确答案区是整块 (49,61,24,36), classify 在它上面
    # 才把 10 个点击判成提交、A1-A4 判成调整(与顺序实验的地面真值一致)。
    # ⚠️只放宽到 GAP=3: 轨道(行18-29)与九宫格(行49-61)相距很远, 不会被误连成
    # 一块。两种粒度的候选**并列**加入, 让后面的排序去选 —— 宁可多一条, 不可漏。
    GAP = 3
    ans: list[tuple[int, int, int, int]] = []
    offsets = {"tight": ((1, 0), (-1, 0), (0, 1), (0, -1)),
               "loose": tuple((dr, dc) for dr in range(-GAP, GAP + 1)
                              for dc in range(-GAP, GAP + 1) if (dr or dc))}
    for nbrs in offsets.values():
        seen = np.zeros_like(mutable, dtype=bool)
        for r0 in range(h):
            for c0 in range(w):
                if not mutable[r0, c0] or seen[r0, c0]:
                    continue
                stack = [(r0, c0)]
                seen[r0, c0] = True
                cells = []
                while stack:
                    r, c = stack.pop()
                    cells.append((r, c))
                    for dr, dc in nbrs:
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < h and 0 <= cc < w and mutable[rr, cc] and not seen[rr, cc]:
                            seen[rr, cc] = True
                            stack.append((rr, cc))
                if len(cells) < 4:
                    continue
                rs = [x for x, _ in cells]
                cs = [y for _, y in cells]
                ans.append((min(rs), max(rs), min(cs), max(cs)))

    ans = list(dict.fromkeys(ans))          # 两种粒度会产生重复, 去掉
    ans.sort(key=lambda b: -((b[1] - b[0] + 1) * (b[3] - b[2] + 1)))

    out: list[RegionMatch] = []
    for a in ans[:4]:
        ah, aw = a[1] - a[0] + 1, a[3] - a[2] + 1
        if ah < 2 or aw < 2 or ah * aw > grid.size // 2:
            continue
        # 题面候选: 同尺寸、有非背景内容、且**整块都改不动**的窗口
        best: list[tuple[int, tuple[int, int, int, int]]] = []
        for r in range(0, h - ah + 1):
            for c in range(0, w - aw + 1):
                if mutable[r:r + ah, c:c + aw].any():
                    continue
                win = grid[r:r + ah, c:c + aw]
                content = int((win != bg).sum())
                if content < 4:
                    continue
                # 🚨题面必须**有图案**, 单色块不算。
                # ls20 实测: harness 把"含钥匙的区域"与**一整块空白**(全是色 3,
                # 而 bg 是别的色)配成了 region_match, 目标于是变成"把钥匙区域
                # 清空" —— 那根本不是过关条件。搜索朝这个错目标努力, h 从 50
                # 降到 3 就再也降不动(八倍算力也不动), 我一度误判成"表征墙"。
                # 单一颜色 = 没有图案 = 不可能是题面。
                if len(set(win.flatten().tolist())) < 2:
                    continue
                best.append((content, (r, r + ah - 1, c, c + aw - 1)))
        best.sort(key=lambda x: -x[0])
        for _, t in best[:3]:
            out.append(RegionMatch(a, t))
            if len(out) >= max_pairs:
                return out
    return out


FAMILIES = {
    "submit_match": SubmitMatch,
    "object_to_object": ObjectToObject,
    "color_count_match": ColorCountMatch,
    "object_reach": ObjectReach,
    "region_match": RegionMatch,
    "color_count": ColorCount,
    "color_appear": ColorAppear,
}


def validate_heuristic(distance, solved_runs: list[list[np.ndarray]]) -> tuple[bool, str]:
    """拿**已通关关卡的完整解**回放, 检验这个启发式到底指不指向目标。

    🚨这道检查是 cd82 L3 用四百多秒算力换来的。当时我给 RegionMatch(下区,
    题面) 当启发式, 它确实有梯度, beam 把它从 100 推到 22 —— 可回头拿 L1/L2
    的解一放才发现: **h 在整个解过程中恒定不变(50/50/50/50), 通关那一步
    反而上升到 90**。那个量跟通关毫无关系, 推它等于白烧算力。

    **有梯度 ≠ 梯度指向目标。** 一个启发式在用于搜索之前, 必须先证明它在
    已知正确的解上是下降的。这是免费的 —— 每通一关就多一条可回放的解。

    🚨判据必须是"**至少有一关上出现过下降**", 不能是"每关都下降"。
    第一版写成"全程恒定即否决", 结果把正确假设误杀了: cd82 的 L1 只需要
    最后一步盖印一次, 而那一步同时通关、画面已切到下一关, 所以在 L1 上
    观察到的 h 全程恒为 50 —— 恒定是**这类游戏的正常现象**, 不是无关的
    证据。同一个假设在 L2 上就有 100→55 的下降。
    一关恒定只说明这关的目标量在最后一步才结算。

    `solved_runs` 每项是一关的逐帧网格序列(含起始帧, 不含通关后的新关卡帧)。
    """
    if not solved_runs:
        return True, "无已通关样本可供检验, 未经验证"
    notes, saw_drop = [], False
    for i, frames in enumerate(solved_runs):
        if len(frames) < 2:
            continue
        hs = [distance(f) for f in frames]
        # 🚨999 是"定位不到对象"的哨兵值, 不是距离。ls20 L2 实锤: 一条假设
        # 45 步真解里 44 步定位不到, 却靠 "999->0" 混过下降判定、拿了
        # "证据 100%" 排到第一名(把全程可定位、真有梯度的那条挤到后面)。
        # 悬崖不是梯度: 大部分帧定位不到 => 这关不提供任何证据。
        loc = [x for x in hs if x < 999]
        if len(loc) * 2 < len(hs):
            notes.append(f"L{i+1}: ⚠️{len(hs)-len(loc)}/{len(hs)} 帧定位不到, 不算证据")
            continue
        if loc and min(loc) < loc[0]:
            saw_drop = True
            notes.append(f"L{i+1}: {loc[0]:.0f}->{min(loc):.0f} 有下降")
        elif len(set(loc)) <= 1:
            notes.append(f"L{i+1}: 全程恒为 {loc[0]:.0f}(可能最后一步才结算)")
        else:
            notes.append(f"L{i+1}: 只升不降 {loc[0]:.0f}->{max(loc):.0f}")
    if saw_drop:
        return True, "; ".join(notes)
    return False, "所有已通关样本上都没出现过下降 —— 与目标无关; " + "; ".join(notes)


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
