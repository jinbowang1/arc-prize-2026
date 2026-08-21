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
    lines = [f"背景色={bg}, 连通块{len(comps)}个(前{min(len(comps), max_items)}):"]
    for n, c, y0, x0, y1, x1 in comps[:max_items]:
        lines.append(f"  色{c} {n}格 行{y0}-{y1} 列{x0}-{x1}")
    return "\n".join(lines)


def _diff_summary(prev: list[list[int]], cur: list[list[int]]) -> str:
    cells = [(y, x) for y in range(len(cur)) for x in range(len(cur[0])) if prev[y][x] != cur[y][x]]
    if not cells:
        return "画面无变化"
    ys = [c[0] for c in cells]; xs = [c[1] for c in cells]
    return f"{len(cells)}格变化, 范围 行{min(ys)}-{max(ys)} 列{min(xs)}-{max(xs)}"


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

        # --- 问 LLM 要计划 ---
        plan: list[Action] = []
        expect = ""
        if llm_calls < max_llm_calls:
            user = (
                f"level {obs.level}/{obs.win_levels}, 已用 {game.steps}/{max_actions} 步, "
                f"可用动作 {list(obs.actions or (1,2,3,4,5,6))}\n\n"
                f"当前画面:\n{_grid_summary(obs.grid)}\n\n"
                f"最近转移:\n" + ("\n".join(recent) or "(无)") + "\n\n"
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
            diff = _diff_summary(prev.grid, obs.grid)
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
