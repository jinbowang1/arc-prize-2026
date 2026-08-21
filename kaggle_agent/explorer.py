"""第一发 agent: 预算内新颖度探索(零模型, 纯符号)。

定位: 打通提交管线的诚实基线, 不是终态方案。策略只有一条原则 ——
**把预算花在历史上更常带来新画面的动作上**(ε-greedy 新颖度)。

三条已在案的经验固化在这里:
- 点击的有效目标是连通块, 不是 4096 个坐标(ft09 定案: 坐标塌缩到 ~20)。
- 过关信号可能滞后(sc25): 每步都查 level/state, 不做"最后一步才查"。
- GAME_OVER 不是失败是信息: RESET 回本关起点继续, 但给肇事动作记死账。
"""
from __future__ import annotations

import random
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Optional

from .remote_env import Action, ApiGame, Obs

GRID_H = GRID_W = 64
MAX_CLICK_TARGETS = 24
EPSILON = 0.12


def _components(grid: list[list[int]]) -> list[tuple[int, int, int]]:
    """4邻域连通块 -> [(size, cy, cx)], 背景色(最大占比色)除外, 大块在前。"""
    h, w = len(grid), len(grid[0])
    counts = Counter(c for row in grid for c in row)
    bg = counts.most_common(1)[0][0]
    seen = [[False] * w for _ in range(h)]
    out: list[tuple[int, int, int]] = []
    for sy in range(h):
        for sx in range(w):
            if seen[sy][sx] or grid[sy][sx] == bg:
                continue
            color = grid[sy][sx]
            q = deque([(sy, sx)])
            seen[sy][sx] = True
            cells = []
            while q:
                y, x = q.popleft()
                cells.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny][nx] and grid[ny][nx] == color:
                        seen[ny][nx] = True
                        q.append((ny, nx))
            cy = sum(c[0] for c in cells) // len(cells)
            cx = sum(c[1] for c in cells) // len(cells)
            # 中心可能落在环形块外 -> 退回块内离中心最近的格
            if grid[cy][cx] != color:
                cy, cx = min(cells, key=lambda c: abs(c[0] - cy) + abs(c[1] - cx))
            out.append((len(cells), cy, cx))
    out.sort(reverse=True)
    return out[:MAX_CLICK_TARGETS]


@dataclass
class _Stat:
    tried: int = 0
    novel: int = 0
    deaths: int = 0

    @property
    def score(self) -> float:
        if self.deaths >= 5 and self.novel == 0:
            return -1.0  # 只会送死的动作
        return (self.novel + 0.5) / (self.tried + 1.0) + (1.0 if self.tried == 0 else 0.0)


@dataclass
class GameResult:
    game_id: str
    levels_completed: int = 0
    win_levels: int = 0
    steps: int = 0
    state: str = "NOT_PLAYED"
    per_level_steps: list[int] = field(default_factory=list)
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return dict(vars(self))


def _key_of(a: Action) -> tuple:
    if not a.data:
        return ("k", a.id)
    p = a.payload
    return ("c", p["x"] // 4, p["y"] // 4)  # 中心量化 4px, 容忍小位移


def play_game(
    game: ApiGame,
    max_actions: int,
    deadline: float,
    rng: Optional[random.Random] = None,
    log=print,
) -> GameResult:
    """探索一个游戏直到 WIN / 动作预算 / 墙钟截止。"""
    rng = rng or random.Random(0)
    res = GameResult(game_id=game.game_id)
    t0 = time.monotonic()
    obs = game.reset()
    res.win_levels = obs.win_levels

    stats: dict[tuple, _Stat] = {}
    seen_hashes: set[int] = set()
    level = obs.level
    level_start = game.steps
    last_key: Optional[tuple] = None

    while game.steps < max_actions and time.monotonic() < deadline:
        if obs.done:
            break
        if obs.dead:
            if last_key is not None:
                stats.setdefault(last_key, _Stat()).deaths += 1
            obs = game.reset_level()
            last_key = None
            continue

        # 换关 = 换动力学: 清账重来
        if obs.level != level:
            res.per_level_steps.append(game.steps - level_start)
            log(f"  [{game.game_id}] level {level}->{obs.level} @ step {game.steps}")
            level, level_start = obs.level, game.steps
            stats.clear()
            seen_hashes.clear()

        avail = obs.actions or (1, 2, 3, 4, 5, 6)
        cands: list[Action] = [Action.key(i) for i in avail if i not in (6, 0)]
        if 6 in avail:
            cands += [Action.click(cx, cy) for _, cy, cx in _components(obs.grid)]
        if not cands:
            cands = [Action.key(i) for i in range(1, 6)]

        if rng.random() < EPSILON:
            a = rng.choice(cands)
        else:
            a = max(cands, key=lambda c: (stats.get(_key_of(c), _Stat()).score, rng.random()))

        prev_hash = hash(str(obs.grid))
        obs = game.act(a)
        last_key = _key_of(a)
        st = stats.setdefault(last_key, _Stat())
        st.tried += 1
        h = hash(str(obs.grid))
        if h not in seen_hashes and h != prev_hash:
            st.novel += 1
        seen_hashes.add(h)

    if obs.done:
        res.per_level_steps.append(game.steps - level_start)
    res.levels_completed = obs.level
    res.steps = game.steps
    res.state = obs.state
    res.seconds = round(time.monotonic() - t0, 1)
    return res
