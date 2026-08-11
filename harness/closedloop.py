"""闭环执行层:边走边比对预测与实测,分歧即反例。

**为什么需要它(2026-08-11 用户人类对照给的批评):**
"你好像一直在用开环的思路在玩,我们人类大多时候是闭环。"

这个批评是对的。cd82 全程我是纯开环 —— 穷举画笔库 → 抽象层规划出完整的
四笔方案 → 真机逐笔执行。开环有个致命性质:**它要求世界模型先完整正确,
否则全盘皆输,而且失败信号出现得极晚**。L4 那次方案在抽象层"解出"了,真机
执行到第四笔才发现无解,而我拿到的信号是"找不到"——一个没有方向的否定。

闭环则是:每一步都拿实测校对预测,**第一次分歧就停**,而且分歧本身指向
"模型里少了什么"。cd82 L4 若是闭环,第一笔涂完就能看见偏差。

**开环还会让人误判证据。** 当时抽象层和真机 beam 都算出"最好差 12 格",
我当成交叉验证 —— 可两条路径吃的是同一份残缺模型,就像两个人拿同一张
少画了一条路的地图,一个开车算一个骑车算,都算出"到不了"。
**同源的两个结论互相印证,等于没印证。**

所以这里的分工是:
  - **开环负责出计划**(抽象层规划便宜,而且能给出全局最少的笔数)
  - **闭环负责执行时校验**,一致就继续,分歧就停
  - 🚨**分歧触发的是"重新采集模型",不是"换个搜索策略"**
    (当时我一看到差 12 就把 beam 从 40 加到 800,方向全错)

这不是新发明 —— ls20 L5 用过 CEGIS(逐步比对预测与真机,分歧当反例修模型),
今天从头到尾没用上,是退步。这个模块就是把它固化下来,免得再忘。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .env import Action, Game, Obs


@dataclass
class Divergence:
    """一次预测与实测的分歧。**这是信息,不是失败。**"""

    step: int
    action: Action
    predicted: np.ndarray
    actual: np.ndarray

    @property
    def cells(self) -> int:
        return int((self.predicted != self.actual).sum())

    @property
    def bbox(self) -> tuple[int, int, int, int] | None:
        ch = np.argwhere(self.predicted != self.actual)
        if len(ch) == 0:
            return None
        return (int(ch[:, 0].min()), int(ch[:, 0].max()),
                int(ch[:, 1].min()), int(ch[:, 1].max()))

    def diagnose(self) -> str:
        """把分歧翻译成"下一步该干什么"。

        默认结论一律指向**重采模型**而不是加大搜索 —— 这是 cd82 L4 用几个
        小时换来的方向。
        """
        b = self.bbox
        if b is None:
            return "无分歧"
        pr = self.predicted
        ac = self.actual
        moved = (pr != ac).sum() > 0 and set(np.unique(pr)) == set(np.unique(ac))
        head = f"第 {self.step} 步 {self.action} 后分歧 {self.cells} 格, 区域 {b}"
        if moved:
            return (head + " —— 颜色集合没变但位置变了, 像是有个**没被跟踪的可动实体**。"
                          "先跑 percept.discover_entities 重新登记画面上有几个东西,"
                          "不要加大搜索。")
        return (head + " —— 实测与模型预测不符, 模型对这个动作的效果记错了。"
                       "重采该动作的效果(必要时换个背景重采,空白背景测不出'擦除'),"
                       "不要加大搜索。")


@dataclass
class LoopResult:
    solved: bool
    seq: list[Action] = field(default_factory=list)
    divergence: Divergence | None = None
    steps_done: int = 0

    def text(self) -> str:
        if self.solved:
            return f"[closedloop] 通关, {len(self.seq)} 步, 全程预测与实测一致"
        if self.divergence:
            return f"[closedloop] 第 {self.steps_done} 步止步 | " + self.divergence.diagnose()
        return f"[closedloop] 计划走完但未通关, {self.steps_done} 步"


def run(game: Game, base: Obs, plan: list[Action], predict,
        region=None, stop_on_divergence: bool = True) -> LoopResult:
    """执行计划,每步比对预测与实测。

    `predict(step_index, obs_before, action) -> np.ndarray | None`
        返回这一步之后**模型认为**画面(或关注区域)应该长什么样;返回 None
        表示模型对这一步没有预测,跳过校验。
    `region(grid) -> np.ndarray` 可选,只校对关注区域(比如答案区),避免被
        计数器之类的无关格子干扰。

    `stop_on_divergence=True` 时第一次分歧就停 —— 这是闭环的要点。继续硬走
    只会让后面的观测都建立在一个已知错误的模型上,越走越没信息。
    """
    obs = base
    done: list[Action] = []
    for i, a in enumerate(plan):
        pred = predict(i, obs, a)
        obs = game.act(a)
        done.append(a)
        if obs.level > base.level:
            return LoopResult(True, done, None, len(done))
        if obs.dead:
            return LoopResult(False, done, None, len(done))
        if pred is None:
            continue
        actual = np.array(obs.grid)
        if region is not None:
            actual = region(actual)
            pred = region(pred) if pred.shape == np.array(obs.grid).shape else pred
        if not np.array_equal(pred, actual):
            div = Divergence(i + 1, a, np.asarray(pred), actual)
            if stop_on_divergence:
                return LoopResult(False, done, div, len(done))
    return LoopResult(False, done, None, len(done))


def layer_sketch(target: np.ndarray, blank: int = 0) -> list[tuple[int, int]]:
    """粗层次假设:先猜"哪个颜色在下层、哪个在上层",再去找实现它的笔。

    人类对照给的另一条(用户原话):"在我脑子里已经能感觉到颜色中哪个是在
    上边,哪个是在下边,我有分层的感觉";"如果能有个大致的层次,会节省很多
    探索次数"。

    我今天的顺序反了 —— 先穷举出 301 种画笔,再规划分层,而最终方案只用到
    4 种。先有层次假设,就能只去找需要的那几支笔。

    启发式很简单:**面积大的颜色多半在底层,面积小的在上层**(上层覆盖下层,
    所以剩下露出来的面积小)。返回 [(颜色, 面积)],从底层到顶层。
    """
    vals, counts = np.unique(target, return_counts=True)
    order = [(int(v), int(c)) for v, c in zip(vals, counts) if v != blank]
    order.sort(key=lambda x: -x[1])
    return order
