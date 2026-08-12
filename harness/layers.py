"""分层渲染模型:画面 = 底图 + 各槽当前选择贡献的层, 按绘制顺序叠加。

**这一块是为了补上世界模型缺的那半边。**

已有两层各解决一半, 但拼不起来:

  - `model.py` 学"一个动作 = 一张固定的覆盖表"。在 r11l L1 成立(23/38 个
    动作), 在 cd82 和 r11l L2 上**全线失效**(可用动作 0) —— 因为那两个
    地方的画面是**组合**决定的, 单个动作只设定其中一维。
  - `factored.py` 测出了组合结构(cd82 L3: 2 个槽 12+4; r11l L2: 2 个槽
    28+8), 但它只给**状态压缩**, 不给**预测能力**: 知道"有两个槽"不等于
    知道"这两个槽选成这样时画面长什么样"。

没有预测能力, 闭环就没有"分歧"可抓 —— 主循环的第④段在这两局上一直是空转的。

**假设(可证伪): 画面 = 底图叠若干层, 每个槽的当前选择贡献一层, 按固定
顺序叠加, 后画盖先画。**

r11l 的结构诊断给了这个假设两条直接支持: 覆盖率 64%(同槽只有最后一次算数
= 一个槽一层)、**交换率 0%**(ab ≠ ba = 确实有前后之分, 不是简单并集)。
用户从 cd82 那次给的话也是这个: "如果能有个大致的层次, 会节省很多探索次数"。

🚨**必须先验证再使用。** 学完要拿留出的组合实测一遍, 报**逐格一致率**和
**整帧一致率**, 不达标就明说建不了模, 交回真机搜。给一个会撒谎的模型比没有
模型更坏 —— 开环规划整个吃它, 而失败信号出现得极晚。
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

import numpy as np

from .env import Action, Game, Obs
from .factored import SlotModel


@dataclass
class LayerModel:
    base: np.ndarray | None = None
    layers: dict[str, dict[tuple[int, int], int]] = field(default_factory=dict)
    slot_of: dict[str, int] = field(default_factory=dict)
    order: list[int] = field(default_factory=list)     # 槽的绘制顺序(先→后)
    cell_acc: float = 0.0        # 留出样本上的逐格一致率
    frame_acc: float = 0.0       # 留出样本上的整帧一致率
    tested: int = 0
    note: str = ""

    @property
    def usable(self) -> bool:
        """够不够格当**预测器**(闭环拿它判分歧)。

        这条必须用整帧一致率。逐格一致率天生虚高 —— 画面绝大部分是背景,
        什么都不预测也有 95%。模型整帧不准就去判分歧, 报出来的全是假分歧,
        闭环会被自己的噪声牵着走。
        """
        return self.tested >= 3 and self.frame_acc >= 0.9

    @property
    def usable_for_ranking(self) -> bool:
        """够不够格当**排序器**(只用来决定先试哪个候选)。

        这条的及格线低得多, 而且理由是实证的: Tycho 的消融里, transition
        模拟精度 16.2% 的配置拿了 88.49 分, 精度 88.1% 的只拿 83.07 分 ——
        **模拟精度不是决定分数的东西**。

        r11l L2 实测: 分层模型整帧一致 0%, 但 166 格的合成里只错 16~24 格
        (12 格是绘制顺序在局部反了, 4~12 格是两层都给不出的真交互)。
        这种模型拿去判分歧是灾难, 拿去给几百个候选排个先后完全够用 ——
        **排错了只是多试几个, 排对了省几百次 fork。正确性从来不归它管。**
        """
        return self.tested >= 3 and self.cell_acc >= 0.98

    def render(self, choices: dict[int, Action]) -> np.ndarray:
        """按绘制顺序叠层。choices = {槽号: 该槽选的动作}。"""
        g = self.base.copy()
        for si in self.order:
            a = choices.get(si)
            if a is None:
                continue
            for (r, c), v in self.layers.get(repr(a), {}).items():
                g[r, c] = v
        return g

    def text(self) -> str:
        tag = "可用" if self.usable else "不可用"
        return (f"[layers] {len(self.layers)} 层 / 绘制顺序 {self.order} | "
                f"留出 {self.tested} 组: 整帧一致 {self.frame_acc:.0%}, "
                f"逐格 {self.cell_acc:.4f} [{tag}] {self.note}")


def _grid(o: Obs) -> np.ndarray:
    return np.array(o.grid)


def learn_layers(game: Game, base: Obs, sm: SlotModel,
                 mask: np.ndarray | None = None,
                 max_pairs: int = 24, holdout: int = 8) -> LayerModel:
    """学层 + 定绘制顺序 + 在留出组合上验收。

    采样很省: 单选样本 = 每个动作一次 fork; 组合样本 = max_pairs 次。
    cd82 L3 规模下总共不到两百次 fork, 几秒钟。
    """
    lm = LayerModel(base=_grid(base))
    slots = [s for s in sm.slots if s] + [[a] for a in sm.loners]
    if len(slots) < 2:
        lm.note = "槽少于两个, 分层没有意义"
        return lm
    for i, s in enumerate(slots):
        for a in s:
            lm.slot_of[repr(a)] = i

    # 一、单选: 每个动作单独在底图上按一次, 变了的格子就是它这一层
    #    ⚠️记的是**结果颜色**不是差量 —— "把某处涂回背景色"也是这一层的
    #    内容, 拿 diff 当效果会把擦除整个漏掉(cd82 画笔那次的老账)。
    for s in slots:
        for a in s:
            o = game.peek(a)
            if o.dead or o.level > base.level:
                continue
            g = _grid(o)
            d = (g != lm.base)
            if mask is not None:
                d &= mask
            lm.layers[repr(a)] = {(int(r), int(c)): int(g[r, c])
                                  for r, c in np.argwhere(d)}

    # 二、定绘制顺序: 拿双选组合实测, 两种顺序各算一次, 谁准用谁
    combos: list[tuple[Action, Action]] = []
    for i, j in itertools.combinations(range(len(slots)), 2):
        for a in slots[i][:4]:
            for b in slots[j][:4]:
                combos.append((a, b))
    combos = combos[:max_pairs + holdout]
    if len(combos) < 3:
        lm.note = "凑不出足够的组合样本"
        return lm

    truth: list[tuple[dict[int, Action], np.ndarray]] = []
    for a, b in combos:
        c = game.fork()
        o = c.act(a)
        if o.dead or o.level > base.level:
            continue
        o = c.act(b)
        if o.dead or o.level > base.level:
            continue
        truth.append(({lm.slot_of[repr(a)]: a, lm.slot_of[repr(b)]: b}, _grid(o)))
    if len(truth) < 3:
        lm.note = "组合样本采不够(多数组合会死或直接过关)"
        return lm

    fit, test = truth[:-holdout] or truth, truth[-holdout:] or truth
    n_slots = len(slots)
    best_order, best_score = None, -1.0
    # 槽不多时枚举全部顺序; 多了就只试正序与逆序, 并把这件事说出来
    orders = (list(itertools.permutations(range(n_slots))) if n_slots <= 4
              else [tuple(range(n_slots)), tuple(reversed(range(n_slots)))])
    if n_slots > 4:
        lm.note = f"槽有 {n_slots} 个, 只试了正序和逆序两种绘制顺序"
    for order in orders:
        lm.order = list(order)
        hit = 0
        for ch, g in fit:
            hit += int(np.array_equal(lm.render(ch), g))
        score = hit / len(fit)
        if score > best_score:
            best_order, best_score = list(order), score
    lm.order = best_order or list(range(n_slots))

    # 三、验收: 只认留出样本
    cells = frames = 0
    total_cells = 0
    for ch, g in test:
        pred = lm.render(ch)
        same = (pred == g)
        if mask is not None:
            same = same | ~mask       # 计数器不算进对错
        cells += int(same.sum())
        total_cells += same.size
        frames += int(bool(same.all()))
    lm.tested = len(test)
    lm.cell_acc = cells / total_cells if total_cells else 0.0
    lm.frame_acc = frames / len(test) if test else 0.0
    return lm


@dataclass
class LayerPlan:
    found: bool
    seq: list[Action] = field(default_factory=list)
    states: int = 0
    seconds: float = 0.0
    best_h: float = float("inf")
    reason: str = ""
    disproved: bool = False   # 真机上把 h 推到 0 了却没过关 = 这条假设被证伪

    def text(self) -> str:
        head = f"分层规划解出 {len(self.seq)} 步" if self.found else f"分层规划未解出({self.reason})"
        kill = " 🚨该假设已被证伪(h 真的到 0 了但没过关)" if self.disproved else ""
        return (f"[layers] {head} | 枚举 {self.states} 个组合, "
                f"h 最好 {self.best_h:.0f}, {self.seconds:.2f}s{kill}")


def plan_on_layers(lm: LayerModel, sm: SlotModel, distance,
                   max_states: int = 200000, max_seconds: float = 30.0) -> LayerPlan:
    """在渲染模型上枚举组合。**每槽最多选一个**, 所以计划长度 = 用到的槽数。

    这是这套表征最实在的好处: 搜索维度从"步数"降到"槽数", 而且步数就是分数。
    渲染一次是纯 numpy, 微秒级 —— 对比真机每扩展一个节点要 fork 一批克隆体
    (r11l 实测 8 次扩展/秒)。
    """
    t0 = time.time()
    slots = [s for s in sm.slots if s] + [[a] for a in sm.loners]
    opts = [[None] + list(s) for s in slots]
    best_h, best_choice = float("inf"), None
    n = 0
    for combo in itertools.product(*opts):
        if n >= max_states or time.time() - t0 > max_seconds:
            return LayerPlan(False, states=n, seconds=time.time() - t0, best_h=best_h,
                             reason=f"枚举到 {n} 个组合就超预算")
        n += 1
        choices = {i: a for i, a in enumerate(combo) if a is not None}
        h = distance(lm.render(choices))
        if h < best_h:
            best_h, best_choice = h, choices
        if h <= 0:
            seq = [choices[i] for i in lm.order if i in choices]
            return LayerPlan(True, seq=seq, states=n, seconds=time.time() - t0, best_h=0.0)

    # 没到 0 也把最好的那个组合交出去 —— 闭环会验它, 分歧本身是信息
    seq = ([best_choice[i] for i in lm.order if i in best_choice]
           if best_choice else [])
    return LayerPlan(False, seq=seq, states=n, seconds=time.time() - t0, best_h=best_h,
                     reason=f"枚举完 {n} 个组合, 最好只到 h={best_h:.0f}")


def plan_and_verify(game: Game, base: Obs, lm: LayerModel, sm: SlotModel,
                    distance, mask: np.ndarray | None = None,
                    top_k: int = 60, max_seconds: float = 60.0) -> LayerPlan:
    """**抽象层排序 + 真机验证**。这是分层模型真正该用的方式。

    分工:
      - 渲染模型只负责**排序**: 把所有组合在 numpy 里渲一遍打分, 微秒级。
        它不准也没关系, 排错了只是多试几个。
      - 真机只负责**判定**: 按排好的顺序逐个在克隆体上实走, 过关只认引擎的
        levels_completed。**模型永远不能宣布通关。**

    这笔账很划算: 组合数 261, 全渲一遍不到 0.1 秒; 每个候选的计划长度 =
    用到的槽数(通常 1~3 步), 在克隆体上验一个只要两三次 fork。前 60 个
    候选全验完约 150 次 fork ≈ 1.5 秒 —— 而盲目 BFS 在同一关上 180 秒才
    扩展一千多个节点。

    ⚠️只验前 top_k 个并把丢掉的个数说出来。静默截断会让"全试过了不行"和
    "根本没试到"看起来一模一样。
    """
    t0 = time.time()
    slots = [s for s in sm.slots if s] + [[a] for a in sm.loners]
    opts = [[None] + list(s) for s in slots]

    scored: list[tuple[float, dict[int, Action]]] = []
    for combo in itertools.product(*opts):
        choices = {i: a for i, a in enumerate(combo) if a is not None}
        if not choices:
            continue
        scored.append((distance(lm.render(choices)), choices))
    scored.sort(key=lambda x: x[0])
    total = len(scored)

    tried = 0
    best_real = float("inf")
    disproved = False
    for h, choices in scored[:top_k]:
        if time.time() - t0 > max_seconds:
            break
        seq = [choices[i] for i in lm.order if i in choices]
        c = game.fork()
        ok = True
        for a in seq:
            o = c.act(a)
            if o.dead:
                ok = False
                break
            if o.level > base.level:
                return LayerPlan(True, seq=seq, states=total,
                                 seconds=time.time() - t0, best_h=0.0)
        tried += 1
        if ok:
            real_h = distance(np.array(o.grid))
            best_real = min(best_real, real_h)
            if real_h <= 0:
                # 🚨**硬否证**: 在真机上把这个量真的推到 0 了, 关卡却没过。
                # 这不是"搜不到", 这是**目标不是它**。
                # r11l L2 实测: 继承自 L1 的 color_count(色 0 的格数达到 74)
                # 就这样被证伪 —— 它本来就是从上一关抄来的常数。
                # 这种否证极其廉价(一次克隆体重放), 而且是**确定**的,
                # 比 validate_heuristic 那种统计性的证据强得多。
                disproved = True

    return LayerPlan(False, states=total, seconds=time.time() - t0, best_h=best_real,
                     disproved=disproved,
                     reason=f"排序后真机验了前 {tried} 个 / 共 {total} 个组合, "
                            f"实测最好 h={best_real:.0f}(未验的 {max(0, total - tried)} 个)")


def predictor(lm: LayerModel, sm: SlotModel):
    """给 closedloop.run 用的 predict(i, obs, a)。

    按计划逐步执行时, 已按下的选择累积在 choices 里, 预测 = 渲染当前组合。
    模型不可用就返回 None —— **校验层要能区分"我预测错了"和"我根本没预测"**。
    """
    state: dict[int, Action] = {}

    def predict(i: int, obs: Obs, a: Action):
        si = lm.slot_of.get(repr(a))
        if si is None or not lm.usable:
            return None
        state[si] = a
        return lm.render(dict(state))

    return predict
