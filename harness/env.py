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

    @property
    def done(self) -> bool:
        return self.level >= self.win_levels

    @property
    def dead(self) -> bool:
        return self.state == "GAME_OVER"


def _obs(frame) -> Obs:
    return Obs(
        grid=frame.frame[-1],
        level=frame.levels_completed,
        win_levels=frame.win_levels,
        state=frame.state.name,
        layers=len(frame.frame),
    )


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
        frame = self._g.perform_action(
            ActionInput(id=_ACTION_BY_ID[a.id], data=a.payload), raw=True
        )
        self.steps += 1
        return _obs(frame)

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
        return child

    def peek(self, a: Action) -> Obs:
        """在克隆体上试一个动作, 不影响本体。真机试探的唯一正确姿势。"""
        return self.fork().act(a)


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
