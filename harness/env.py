"""环境接入与安全分叉。

三条铁律固化在这里(来自 ls20/tr87/ft09 实测, 不要"优化"掉):

1. 分叉只用「整对象单 memo deepcopy」。部分快照必致状态泄漏——ls20 上实测
   能量条 region 漏过, 且 game 对象里 ui/camera 与 sprite 有交叉引用、
   动态模块类 pickle 不进去。微优化到此为止。
2. `_clean_levels`(静态关卡数据, ~120KB)在分叉前摘出、分叉后共享回去。
   它是只读的, 不需要复制, 复制它会让分叉慢一个量级。
3. OPERATION_MODE=OFFLINE 必须在 import arc_agi 之前设好, 否则公司网墙下
   每次 make 都在等 arcprize SSL 超时(3 分钟 vs 秒回)。
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Any, Iterator

os.environ.setdefault("OPERATION_MODE", "OFFLINE")

import numpy as np  # noqa: E402
import arc_agi  # noqa: E402  (必须在 OPERATION_MODE 之后)
from arcengine import ActionInput, GameAction  # noqa: E402

_ACTION_BY_ID = {i: getattr(GameAction, f"ACTION{i}") for i in range(1, 8)}


@dataclass(frozen=True)
class Action:
    """统一动作表示: keyboard 是 (id, {}), click 是 (6, {"x":.., "y":..})。"""

    id: int
    data: tuple[tuple[str, int], ...] = ()

    @classmethod
    def key(cls, i: int) -> "Action":
        return cls(i)

    @classmethod
    def click(cls, x: int, y: int, action_id: int = 6) -> "Action":
        return cls(action_id, (("x", x), ("y", y)))

    @property
    def payload(self) -> dict[str, int]:
        return dict(self.data)

    def __repr__(self) -> str:
        if not self.data:
            return f"A{self.id}"
        return f"A{self.id}({self.payload['x']},{self.payload['y']})"


@dataclass
class Obs:
    """一次动作后的观测。grid 是 64x64 的颜色索引二维列表(取最后一层)。

    `layers` = 引擎这次返回了几帧。>1 说明这一步含动画序列, 中间层是动画帧
    不是决策帧 —— 拿动画帧做 diff 会得出错误的动力学。
    """

    grid: list[list[int]]
    level: int
    win_levels: int
    state: str
    layers: int = 1
    actions: tuple[int, ...] = ()   # 引擎自报的可用动作 id, 别在上层写死
    pending: int = 0                # 已注入但**尚未结算**的动作 id(0=无), 见 Game.act

    @property
    def done(self) -> bool:
        return self.level >= self.win_levels

    @property
    def dead(self) -> bool:
        return self.state == "GAME_OVER"


def _obs(frame, pending: int = 0) -> Obs:
    return Obs(
        pending=pending,
        grid=frame.frame[-1],
        level=frame.levels_completed,
        win_levels=frame.win_levels,
        state=frame.state.name,
        layers=len(frame.frame),
        actions=tuple(sorted(_action_id(a) for a in (frame.available_actions or []))),
    )


def _action_id(a) -> int:
    """available_actions 里可能是 GameAction 枚举也可能已是 int。"""
    if isinstance(a, int):
        return a
    v = getattr(a, "value", None)
    if isinstance(v, int):
        return v
    return int(str(getattr(a, "name", a)).replace("ACTION", ""))


class Game:
    """真机或克隆体的统一句柄。

    真机(`Game.make`)推进的步数计入官方 scorecard;克隆体(`fork`)完全免费,
    是三局攻关的核心杠杆——试错不烧分母。
    """

    def __init__(self, raw_game: Any, game_id: str, is_fork: bool = False):
        self._g = raw_game
        self.game_id = game_id
        self.is_fork = is_fork
        self.steps = 0
        self._pending = 0
        self.lagged = False        # 见 detect_lag: 动作是否要下一次调用才结算

    @classmethod
    def make(cls, game_id: str) -> tuple["Game", Obs]:
        arcade = arc_agi.Arcade()
        env = arcade.make(game_id)
        if env is None:
            raise RuntimeError(f"make({game_id}) 返回 None")
        frame = env.reset()
        game = cls(env._game, game_id)
        game._env = env  # 保留, scorecard 对账要用
        return game, _obs(frame)

    def act(self, a: Action) -> Obs:
        """走一步。

        🚨**返回的画面未必包含这一步的效果。** sc25 实测: `perform_action` 只把
        动作放进输入缓冲, 引擎的 `step()` 才结算, 而官方 SDK 的 `Environment.step`
        内部同样只调 `perform_action` —— 所以这是**游戏的规则, 不是 harness 的
        bug**, 不能靠偷调 `step()` 绕过去(那样算出的解在官方接口下不成立)。

        后果是隐藏状态: 走 A3 和走 A1 之后画面**完全相同**(都还是开局), 只有
        "缓冲里待结算的是谁"不同。指纹只看画面就会把它们当成同一个状态,
        全部去重 -> BFS 深度 0 就报"队列穷尽"(sc25 裸跑 1 秒结束就是这么来的)。
        所以把待结算动作记进 `Obs.pending`, 让指纹带上它。

        ⚠️与 ls20/tr87/ft09/cd82 不冲突: 那四局动作即时生效, pending 只是多一个
        恒定维度, 四局在案解重放回归照常通过。
        """
        frame = self._g.perform_action(
            ActionInput(id=_ACTION_BY_ID[a.id], data=a.payload), raw=True
        )
        self.steps += 1
        self._pending = a.id
        return _obs(frame, a.id)

    def replay(self, seq: list[Action]) -> Obs:
        obs = None
        for a in seq:
            obs = self.act(a)
        return obs

    def fork(self) -> "Game":
        """整对象单 memo deepcopy(~10ms)。慢而正确, 别改。"""
        clean = getattr(self._g, "_clean_levels", None)
        self._g._clean_levels = None
        try:
            copied = copy.deepcopy(self._g)
        finally:
            self._g._clean_levels = clean
        copied._clean_levels = clean
        child = Game(copied, self.game_id, is_fork=True)
        child.steps = self.steps
        child._pending = self._pending      # 缓冲里待结算的动作也要跟着克隆
        child.lagged = self.lagged
        return child

    def peek(self, a: Action) -> Obs:
        """在克隆体上试一个动作, 不影响本体。真机试探的唯一正确姿势。"""
        return self.fork().act(a)

    def detect_lag(self, acts: list[Action]) -> bool:
        """探测这一局是不是"动作要下一次调用才结算"。就地记在 self.lagged。

        判据: **单步 peek 全都改 0 格, 而走两次有变化**。两个条件都要 ——
        只看前者会把"开局恰好动不了"误判成滞后。

        sc25 实测 27 个动作单步 peek **全部**改 0 格, 走两次 14 个有效;
        ls20/tr87/ft09/cd82 动作即时生效, 这里返回 False, 表征层照旧走单步。
        """
        base = self._grid()
        single_any = double_any = False
        for a in acts[:8]:
            o1 = self.peek(a)
            if not o1.dead and not np.array_equal(np.array(o1.grid), base):
                single_any = True
                break
            n = self.fork()
            oa = n.act(a)
            if oa.dead:
                continue
            ob = n.act(a)
            if not ob.dead and not np.array_equal(np.array(ob.grid), base):
                double_any = True
        self.lagged = (not single_any) and double_any
        return self.lagged

    def _grid(self) -> "np.ndarray":
        return np.array(self._g.get_pixels(0, 0, 64, 64))

    def effect(self, a: Action) -> Obs:
        """看清 a 的效果。滞后局走两次, 其余等同 peek。

        🚨为什么需要它: sc25 的 `perform_action` 只注入、`step()` 才结算(见
        `act` 的注释), 所以单步 peek 看到的是**上一个**动作的效果。实测后果 ——
        27 个动作单步全报"改 0 格" -> 实体发现 **0 个** / model **可用 0** /
        抽象层建不出覆盖表, 整个表征层是瞎的; 而 best_first 的 h 卡在 3,
        算力加 5.2 倍一格不动(表征墙, 不是搜索墙)。

        走两次 a 得到的正是 a 的**单次真效果**(实测 viaact(3,3) 逐格等于
        truth(3)), 且克隆体试探免费 —— 不必偷调 `step()` 脱离官方语义。

        ⚠️返回的状态是"a 已结算一次 + 缓冲里还有一个 a", **不是**"走了一次 a"。
        所以只能用来**观测效果**(建表征), 不能当搜索的后继状态 —— 搜索走 act
        语义, 指纹带 pending 就够(见 search.fingerprint)。
        """
        if not getattr(self, "lagged", False):
            return self.peek(a)
        n = self.fork()
        o = n.act(a)
        return o if o.dead else n.act(a)


def action_space(obs_actions: list[int], grid_h: int = 64, grid_w: int = 64) -> dict:
    """把 available_actions 翻译成 harness 内部的动作空间描述。

    click 动作(6/7)的原始空间是 grid_h*grid_w, 但 ft09 定案: 有效点击目标是
    连通块(4096 -> ~20)。这里只报告原始空间, 塌缩交给 percept 层。
    """
    keys = [i for i in obs_actions if i <= 5]
    clicks = [i for i in obs_actions if i >= 6]
    return {
        "keys": keys,
        "clicks": clicks,
        "raw_click_space": grid_h * grid_w if clicks else 0,
        "kind": ("keyboard_click" if keys and clicks else "click" if clicks else "keyboard"),
    }
