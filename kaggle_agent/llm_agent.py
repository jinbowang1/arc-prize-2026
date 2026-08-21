"""LLM-agent 主循环(第二阶段, v2 提交候选)。

范式 = Milestone1 季军(Reki/forge)一路: 结构化观察 -> LLM 出 JSON 计划 +
维护"规则笔记"(反思记忆) -> 执行 -> 把真实 diff 反馈回去。外加两条我们
自己的纪律:

- **分歧即停**: LLM 每次计划都附带预期(expect); 执行中画面 diff 与预期
  明显不符就中断剩余计划, 把"预期 vs 实际"喂回下一轮 —— 计划便宜,
  在错误世界模型上走完计划才贵(每步都计 RHAE 分母)。
- **坏输出不崩局**: 模型输出随时可能不合法。解析失败/调用失败 -> 降级成
  新颖度探索走几步, 循环继续。LLM 是增强, 不是单点故障。

⚠️与 explorer 同一条 API-only 纪律: 没有克隆体, 每个动作都计分。
"""
from __future__ import annotations

import random
import time
from collections import Counter, deque
from typing import Optional

from .explorer import GameResult, _components
from .llm import parse_json_block
from .remote_env import Action, ApiGame, Obs

MAX_PLAN_LEN = 8          # 单轮最多执行的计划步数(防模型一口气梭哈)
FALLBACK_STEPS = 4        # LLM 失效时用探索垫的步数
RECENT_TRANSITIONS = 12   # 喂给模型的近期转移条数

SYSTEM_PROMPT = """你在玩一个 64x64 网格解谜游戏, 目标是尽快过关(level 上升)。
你看不到说明书, 必须从动作效果里归纳规则。每个动作都有成本, 越省越好。
动作: A1-A5 是按键(语义未知, 要试), A6(x,y) 是点击坐标。
每轮你会收到: 当前画面的连通块摘要 / 最近动作与画面变化 / 你自己维护的规则笔记。
严格输出一个 JSON 对象:
{"notes": "更新后的完整规则笔记(会原样带到下一轮, 精炼准确)",
 "plan": [{"a": 1} 或 {"a": 6, "x": 30, "y": 40}, ...],
 "expect": "执行这个计划后画面预期发生什么(一句话)"}
计划最多 8 步。若还在摸索, 就出 1-2 步的试探计划并在 notes 里记下待验证假设。"""


def _grid_summary(grid: list[list[int]], max_items: int = 36) -> str:
    """连通块摘要: 颜色/大小/包围盒。对 LLM 来说比 4096 个数字有用得多。"""
    h, w = len(grid), len(grid[0])
    counts = Counter(c for row in grid for c in row)
    bg = counts.most_common(1)[0][0]
    seen = [[False] * w for _ in range(h)]
    comps = []
    for sy in range(h):
        for sx in range(w):
            if seen[sy][sx] or grid[sy][sx] == bg:
                continue
            color = grid[sy][sx]
            q = deque([(sy, sx)])
            seen[sy][sx] = True
            ys, xs, n = [sy], [sx], 1
            while q:
                y, x = q.popleft()
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny][nx] and grid[ny][nx] == color:
                        seen[ny][nx] = True
                        q.append((ny, nx))
                        ys.append(ny); xs.append(nx); n += 1
            comps.append((n, color, min(ys), min(xs), max(ys), max(xs)))
    comps.sort(reverse=True)
    lines = [f"全景(16x16 降采样, 每格=原图4x4):\n{_overview(grid)}\n",
             f"背景色={bg}, 连通块{len(comps)}个(前{min(len(comps), max_items)}):"]
    patches = 0
    for n, c, y0, x0, y1, x1 in comps[:max_items]:
        line = f"  色{c} {n}格 行{y0}-{y1} 列{x0}-{x1}"
        # 小块(图案候选)给像素级视图 —— 图案往往就是题眼
        if patches < 10 and (y1 - y0) < 14 and (x1 - x0) < 14:
            line += "\n" + "\n".join("    " + r for r in
                                     _render_patch(grid, y0, x0, y1, x1).split("\n"))
            patches += 1
        lines.append(line)
    return "\n".join(lines)


def _render_patch(grid: list[list[int]], y0: int, x0: int, y1: int, x1: int) -> str:
    """把一小块像素渲染成十六进制字符行(0-f=色号16以内, 其余用'?')。

    连通块摘要会抹掉图案本身(ft09 探针实测: 蓝图 3x3 花纹就是答案, 模型
    看不见图案等于蒙眼玩拼图), 小块必须给像素级视图。
    """
    rows = []
    for y in range(y0, y1 + 1):
        rows.append("".join(format(grid[y][x], "x") if grid[y][x] < 16 else "?"
                            for x in range(x0, x1 + 1)))
    return "\n".join(rows)


def _overview(grid: list[list[int]], cell: int = 4) -> str:
    """64x64 -> 16x16 众数降采样全景图, 让模型有整体布局感。"""
    h, w = len(grid), len(grid[0])
    rows = []
    for by in range(0, h, cell):
        row = []
        for bx in range(0, w, cell):
            c = Counter(grid[y][x] for y in range(by, min(by + cell, h))
                        for x in range(bx, min(bx + cell, w))).most_common(1)[0][0]
            row.append(format(c, "x") if c < 16 else "?")
        rows.append("".join(row))
    return "\n".join(rows)


def _diff_summary(prev: list[list[int]], cur: list[list[int]], detail: bool = True,
                  ignore: frozenset = frozenset()) -> str:
    cells = [(y, x) for y in range(len(cur)) for x in range(len(cur[0]))
             if prev[y][x] != cur[y][x] and (y, x) not in ignore]
    if not cells:
        return "画面无变化"
    ys = [c[0] for c in cells]; xs = [c[1] for c in cells]
    s = f"{len(cells)}格变化, 范围 行{min(ys)}-{max(ys)} 列{min(xs)}-{max(xs)}"
    if detail and (max(ys) - min(ys)) < 14 and (max(xs) - min(xs)) < 14:
        s += ("; 变化区现状:\n" + _render_patch(cur, min(ys), min(xs), max(ys), max(xs)))
    return s


def _probe_actions(game: ApiGame, obs: Obs) -> tuple[Obs, str]:
    """关卡开局把每个按键各走一步, 产出动作效果表。

    每关最多花 5 个计分动作换一张效果表, 比让模型一轮一轮花 LLM 调用去
    试便宜得多(主动学习效率=胜负手)。⚠️动作真实计分, 死了就地重置并记录。
    """
    lines = []
    for i in [a for a in (obs.actions or ()) if a not in (0, 6)][:5]:
        prev = obs
        obs = game.act(Action.key(i))
        if obs.dead:
            lines.append(f"A{i}: 导致死亡(已重置)")
            obs = game.reset_level()
            continue
        lines.append(f"A{i}: {_diff_summary(prev.grid, obs.grid, detail=False)}")
        if obs.done or obs.level != prev.level:
            lines.append(f"A{i} 直接导致过关!")
            break
    return obs, "\n".join(lines) or "(无按键动作可探测)"


class _Volatile:
    """跨步计数器探测(harness 跨步判据的轻量版)。

    每步都自动变化的格子(步数条/时间条)会把 diff 污染成噪声 —— r11l 列0
    竖条实测让模型把注意力全耗在假信号上。判据: 观测≥4次转移后, 变化率
    ≥80% 的格子进掩码, diff 摘要里剔除。
    """

    def __init__(self):
        self.counts: Counter = Counter()
        self.total = 0

    def update(self, prev: list[list[int]], cur: list[list[int]]) -> None:
        self.total += 1
        for y in range(len(cur)):
            for x in range(len(cur[0])):
                if prev[y][x] != cur[y][x]:
                    self.counts[(y, x)] += 1

    @property
    def mask(self) -> frozenset:
        if self.total < 4:
            return frozenset()
        return frozenset(c for c, n in self.counts.items() if n / self.total >= 0.8)


def _parse_plan(d: dict, avail: tuple[int, ...]) -> list[Action]:
    out: list[Action] = []
    for step in (d.get("plan") or [])[:MAX_PLAN_LEN]:
        try:
            a = int(step["a"])
            if a == 6:
                x, y = int(step["x"]), int(step["y"])
                if 6 in avail and 0 <= x < 64 and 0 <= y < 64:
                    out.append(Action.click(x, y))
            elif a in avail:
                out.append(Action.key(a))
        except (KeyError, TypeError, ValueError):
            continue  # 坏步跳过, 好步保留
    return out


def _fallback_action(obs: Obs, rng: random.Random) -> Action:
    avail = obs.actions or (1, 2, 3, 4, 5, 6)
    cands = [Action.key(i) for i in avail if i not in (0, 6)]
    if 6 in avail:
        cands += [Action.click(cx, cy) for _, cy, cx in _components(obs.grid)[:8]]
    return rng.choice(cands) if cands else Action.key(1)


def play_game_llm(
    game: ApiGame,
    llm,
    max_actions: int,
    deadline: float,
    max_llm_calls: int = 40,
    rng: Optional[random.Random] = None,
    log=print,
) -> GameResult:
    rng = rng or random.Random(0)
    res = GameResult(game_id=game.game_id)
    t0 = time.monotonic()
    obs = game.reset()
    res.win_levels = obs.win_levels

    notes = "(还没有笔记)"
    recent: deque[str] = deque(maxlen=RECENT_TRANSITIONS)
    level, level_start = obs.level, game.steps
    llm_calls = 0
    feedback = ""   # 上一轮"预期 vs 实际"的对账, 喂回下一轮
    vol = _Volatile()
    obs, probe_table = _probe_actions(game, obs)

    while game.steps < max_actions and time.monotonic() < deadline and not obs.done:
        if obs.dead:
            recent.append("(死亡, 已重置回本关起点)")
            obs = game.reset_level()
            continue
        if obs.level != level:
            res.per_level_steps.append(game.steps - level_start)
            log(f"  [{game.game_id}] level {level}->{obs.level} @ step {game.steps}")
            level, level_start = obs.level, game.steps
            recent.append(f"(过关! 现在是 level {obs.level})")
            obs, probe_table = _probe_actions(game, obs)  # 新关重探(机制常换)

        # --- 问 LLM 要计划 ---
        plan: list[Action] = []
        expect = ""
        if llm_calls < max_llm_calls:
            user = (
                f"level {obs.level}/{obs.win_levels}, 已用 {game.steps}/{max_actions} 步, "
                f"可用动作 {list(obs.actions or (1,2,3,4,5,6))}\n\n"
                f"当前画面:\n{_grid_summary(obs.grid)}\n\n"
                f"本关按键效果表(开局各试一次):\n{probe_table}\n\n"
                + (f"(已自动忽略每步都自变的{len(vol.mask)}个格子, 疑似步数条/计时器)\n\n"
                   if vol.mask else "")
                + f"最近转移:\n" + ("\n".join(recent) or "(无)") + "\n\n"
                f"你的规则笔记:\n{notes}\n" + (f"\n上轮对账: {feedback}\n" if feedback else "")
            )
            try:
                raw = llm.chat([{"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": user}])
                llm_calls += 1
                d = parse_json_block(raw)
                if d.get("notes"):
                    notes = str(d["notes"])[:4000]
                expect = str(d.get("expect", ""))[:300]
                plan = _parse_plan(d, obs.actions or (1, 2, 3, 4, 5, 6))
            except Exception as e:  # noqa: BLE001
                log(f"  [{game.game_id}] LLM 失效, 降级探索: {e!r}")

        if not plan:  # LLM 失效/出空计划 -> 探索垫几步, 攒新观察再问
            plan = [_fallback_action(obs, rng) for _ in range(FALLBACK_STEPS)]
            expect = ""

        # --- 执行 + 分歧即停 ---
        any_change = False
        for i, a in enumerate(plan):
            if game.steps >= max_actions or time.monotonic() >= deadline:
                break
            prev = obs
            obs = game.act(a)
            vol.update(prev.grid, obs.grid)
            diff = _diff_summary(prev.grid, obs.grid, ignore=vol.mask)
            recent.append(f"step{game.steps}: {a} -> {diff}")
            if diff != "画面无变化":
                any_change = True
            if obs.done or obs.dead or obs.level != level:
                break
            # 分歧判据(保守): 预期有变化但连续两步纹丝不动 -> 计划建立在错的
            # 世界模型上, 停下回炉, 别把剩余步数烧完
            if expect and not any_change and i >= 1:
                feedback = f"预期「{expect}」, 但前{i+1}步画面均无变化, 计划已中断"
                break
        else:
            feedback = f"计划执行完毕(预期「{expect}」)" if expect else ""

    if obs.done:
        res.per_level_steps.append(game.steps - level_start)
    res.levels_completed = obs.level
    res.steps = game.steps
    res.state = obs.state
    res.seconds = round(time.monotonic() - t0, 1)
    return res
