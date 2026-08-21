"""API-only 环境句柄。

铁律: 只用 EnvironmentWrapper.reset()/step()。没有 fork。
每次 act/reset_level 都计入 steps —— 这就是 RHAE 的分母, 记账必须诚实。

ONLY_RESET_LEVELS=true 时 RESET 是关卡重置(回本关起点), 且自身计 1 步
(ls20 实测: 20浪费+RESET+13=34 步, score=(22/34)^2 精确吻合)。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from arcengine.enums import GameAction

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
    """一次动作后的观测。grid 是 64x64 颜色索引二维列表(取最后一层)。

    ⚠️画面未必包含刚才那步的效果(sc25: perform_action 只注入, 下次调用才
    结算)。API-only 模式下没有克隆体可以做 detect_lag, 探索策略要靠
    "新颖度按动作序列记账"天然容忍一拍滞后, 不要假设即时生效。
    """

    grid: list[list[int]]
    level: int
    win_levels: int
    state: str
    layers: int = 1
    actions: tuple[int, ...] = ()

    @property
    def done(self) -> bool:
        return self.state == "WIN" or (self.win_levels > 0 and self.level >= self.win_levels)

    @property
    def dead(self) -> bool:
        return self.state == "GAME_OVER"


def _action_id(a: Any) -> int:
    if isinstance(a, int):
        return a
    v = getattr(a, "value", None)
    if isinstance(v, int):
        return v
    return int(str(getattr(a, "name", a)).replace("ACTION", ""))


def _obs(frame: Any) -> Obs:
    if frame is None:
        raise RuntimeError("step() 返回 None(网关拒绝或游戏未就绪)")
    return Obs(
        grid=frame.frame[-1],
        level=frame.levels_completed,
        win_levels=frame.win_levels,
        state=frame.state.name,
        layers=len(frame.frame),
        actions=tuple(sorted(_action_id(a) for a in (frame.available_actions or []))),
    )


class ApiGame:
    """一个游戏的句柄。steps = 已计分动作数(RESET 也算)。"""

    def __init__(self, env: Any, game_id: str):
        self._env = env
        self.game_id = game_id
        self.steps = 0

    def reset(self) -> Obs:
        """开局 reset。⚠️只在游戏开始时调一次, 之后的 RESET 走 reset_level。"""
        return _obs(self._env.reset())

    def act(self, a: Action) -> Obs:
        frame = self._env.step(_ACTION_BY_ID[a.id], a.payload or None)
        self.steps += 1
        return _obs(frame)

    def reset_level(self) -> Obs:
        """死后重来(ONLY_RESET_LEVELS 下回本关起点)。计 1 步。"""
        frame = self._env.step(GameAction.RESET)
        self.steps += 1
        return _obs(frame)
