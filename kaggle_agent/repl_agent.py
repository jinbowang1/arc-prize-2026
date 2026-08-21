"""REPL-agent (v3): 给 LLM 一只能动手的手。

探针实测钉死的结论: 对话式"看-猜-试"里模型在口述科学 —— 提得出假设
(r11l"变化行下界递增")却没法验证, 只能在思考里空转到 token 耗尽。
Duck(Milestone1 冠军)的核心武器就是这只手: 把游戏状态变成 Python 变量,
让模型写代码去算、去验、去动。

协议:
- 模型每轮输出一个 ```python 代码块, 我们在持久命名空间里 exec,
  stdout/异常截断后喂回, 模型据此迭代。
- 命名空间预装: grid(当前画面) / history(全部转移) / act()(真机动作, 计分)
  / components()/diff() 分析函数 / notes(模型自己维护的持久笔记)。
- 上下文用消息驱逐(Duck 同款): 系统提示+开局观察+最近 K 轮, notes 每轮重注入。

纪律:
- act() 是唯一花钱的东西, 代码里算什么都免费 —— 这正是要教给模型的经济学。
- exec 有墙钟(防死循环)和输出截断(防刷屏)。
- 过关/死亡由 act() 内部广播, 预算耗尽抛 BudgetExhausted 终止本局。
"""
from __future__ import annotations

import io
import contextlib
import signal
import time
import traceback
from collections import deque

from .explorer import GameResult
from .llm_agent import _grid_summary
from .remote_env import Action, ApiGame, Obs

EXEC_TIMEOUT_S = 30
MAX_OUTPUT_CHARS = 3000
KEEP_EXCHANGES = 6

SYSTEM_PROMPT = """你在玩一个 64x64 网格解谜游戏(颜色索引 0-15), 目标是尽快让 level 上升直到通关。
没有说明书, 规则要自己从实验里归纳。你通过写 Python 代码来观察、分析和行动。

每轮你输出一个 ```python 代码块(只执行第一个), 我会执行并把输出返回给你。

可用的变量与函数:
- grid: 当前画面, 64x64 的 list[list[int]]
- level, win_levels, steps_used, steps_budget: 进度与预算
- act(a, x=None, y=None): 执行真实动作并更新 grid/history。a=1..5 是按键, a=6 是点击(要 x,y; x=列, y=行)。
  返回本次动作改变的格子数。**每次 act 都消耗预算且不可撤销, 省着用!**
- history: 已发生的全部转移 [(动作描述, before_grid, after_grid), ...] —— 验证假设先在这上面算, 免费!
- components(g): 连通块列表 [(size, color, y0, x0, y1, x1), ...]
- diff(g1, g2): 两画面不同的格子 [(y, x, 旧值, 新值), ...]
- patch(g, y0, x0, y1, x1): 把子区域渲染成字符串(每格一个十六进制色号)
- notes: 一个 dict, 你的持久笔记本(跨轮保留), 把归纳出的规则、待验证假设记在里面

- verify_wm(predict): 把你写的世界模型 predict(before_grid, action_str)->after_grid
  在全部 history 上逐格重放验证, 返回每条错格数。**全 0 才算规则成立。**

工作流程(按顺序):
1. 探索: 花预算的 20%-40% 做实验摸动作效果(太少看不懂规则, 太多烧掉分数), 用 patch/diff 看清每次改了什么;
2. 验证: 把猜出的规则写成 predict 函数, 用 verify_wm 验证 —— **没有通过 verify_wm 的规则不许用来规划**(经验: 不验证的世界模型比没有世界模型还糟);
3. 简化: 规则验证通过后, 试着把它改写得更简单(合并分支/去掉特例), 再验一次 —— 更简单且同样全对的规则泛化更好;
4. 规划: 在验证过的规则上写代码推演出过关序列, 再用最少的 act 执行;
5. 复杂关卡可以只对"关键子系统"建模(比如只预测被点击区域), 但验证不能省。
你在代码里定义的函数和变量会跨轮保留 —— 把好用的分析工具、验证过的 predict 沉淀下来复用。
print 出你想看的东西; 输出会截断到 3000 字符, 别打印整个 grid(用 patch 看局部)。"""


class BudgetExhausted(Exception):
    pass


class LevelUp(Exception):
    pass


def _extract_code(text: str) -> str | None:
    if "```" not in text:
        return None
    for part in text.split("```")[1:]:
        body = part
        if body.startswith("python"):
            body = body[6:]
        elif body.startswith("py"):
            body = body[2:]
        body = body.strip("\n")
        if body.strip():
            return body
    return None


def _components(g):
    from collections import Counter as C, deque as D
    h, w = len(g), len(g[0])
    bg = C(c for row in g for c in row).most_common(1)[0][0]
    seen = [[False] * w for _ in range(h)]
    out = []
    for sy in range(h):
        for sx in range(w):
            if seen[sy][sx] or g[sy][sx] == bg:
                continue
            color = g[sy][sx]
            q = D([(sy, sx)])
            seen[sy][sx] = True
            ys, xs, n = [sy], [sx], 1
            while q:
                y, x = q.popleft()
                for ny, nx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny][nx] and g[ny][nx] == color:
                        seen[ny][nx] = True
                        q.append((ny, nx))
                        ys.append(ny); xs.append(nx); n += 1
            out.append((n, color, min(ys), min(xs), max(ys), max(xs)))
    out.sort(reverse=True)
    return out


def _diff(g1, g2):
    return [(y, x, g1[y][x], g2[y][x]) for y in range(len(g2))
            for x in range(len(g2[0])) if g1[y][x] != g2[y][x]]


def _patch(g, y0, x0, y1, x1):
    return "\n".join("".join(format(g[y][x], "x") if g[y][x] < 16 else "?"
                             for x in range(x0, x1 + 1)) for y in range(y0, y1 + 1))


class _TimeoutError(Exception):
    pass


def _alarm(signum, frame):  # noqa: ARG001
    raise _TimeoutError()


def play_game_repl(
    game: ApiGame,
    llm,
    max_actions: int,
    deadline: float,
    max_rounds: int = 30,
    log=print,
) -> GameResult:
    res = GameResult(game_id=game.game_id)
    t0 = time.monotonic()
    obs = game.reset()
    res.win_levels = obs.win_levels
    state = {"obs": obs, "level_start": game.steps}

    def act(a, x=None, y=None):
        if game.steps >= max_actions or time.monotonic() >= deadline:
            raise BudgetExhausted("动作预算或时间已耗尽")
        action = Action.click(int(x), int(y)) if int(a) == 6 else Action.key(int(a))
        prev = state["obs"]
        cur = game.act(action)
        if cur.dead:
            ns["history"].append((repr(action), prev.grid, cur.grid))
            cur = game.reset_level()
            state["obs"] = cur
            _sync()
            print(f"[{action} 导致死亡, 已重置回本关起点]")
            return -1
        ns["history"].append((repr(action), prev.grid, cur.grid))
        state["obs"] = cur
        _sync()
        if cur.done or cur.level != prev.level:
            raise LevelUp()
        return len(_diff(prev.grid, cur.grid))

    def verify_wm(predict):
        """世界模型验证器(Tycho 纪律): 把你写的 predict(before_grid, action_str)
        -> after_grid 在 history 全部转移上逐格重放, 返回每条的错格数。
        全 0 才算世界模型成立; 成立之后你就能在脑外免费推演了。"""
        report = []
        for act_str, before, after in ns["history"]:
            try:
                pred = predict(before, act_str)
                bad = sum(1 for y in range(len(after)) for x in range(len(after[0]))
                          if pred[y][x] != after[y][x])
            except Exception as e:  # noqa: BLE001
                bad = f"异常:{e!r}"
            report.append((act_str, bad))
        ok = sum(1 for _, b in report if b == 0)
        print(f"verify_wm: {ok}/{len(report)} 条转移逐格全对")
        return report

    ns: dict = {
        "act": act, "history": [], "components": _components, "diff": _diff,
        "patch": _patch, "notes": {}, "verify_wm": verify_wm,
    }

    def _sync():
        o = state["obs"]
        ns.update(grid=o.grid, level=o.level, win_levels=o.win_levels,
                  steps_used=game.steps, steps_budget=max_actions)

    _sync()
    # exchanges 存 (喂回的user消息, 模型的assistant输出); 组装时保持
    # system, first_user, a1, u1, a2, u2... 的严格交替 —— 最新结果只出现一次
    exchanges: deque[tuple[str, str]] = deque(maxlen=KEEP_EXCHANGES)
    first_user = (f"开局: level {obs.level}/{obs.win_levels}, 预算 {max_actions} 动作, "
                  f"可用动作 {list(obs.actions or (1,2,3,4,5,6))}\n\n{_grid_summary(obs.grid)}")

    for rnd in range(max_rounds):
        if time.monotonic() >= deadline or game.steps >= max_actions or state["obs"].done:
            break
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": first_user}]
        for u, a in exchanges:
            msgs += [{"role": "assistant", "content": a}, {"role": "user", "content": u}]
        try:
            raw = llm.chat(msgs)
        except Exception as e:  # noqa: BLE001
            log(f"  [{game.game_id}] LLM 失效: {e!r}")
            break
        code = _extract_code(raw)
        if not code:
            status = (f"[round {rnd+1}/{max_rounds}] level {state['obs'].level}/{res.win_levels}, "
                      f"已用 {game.steps}/{max_actions} 动作")
            exchanges.append(("(没有找到 ```python 代码块。请只输出一个代码块。)\n" + status,
                              raw[-1500:]))
            continue

        buf = io.StringIO()
        outcome = ""
        old = signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(EXEC_TIMEOUT_S)
        try:
            with contextlib.redirect_stdout(buf):
                exec(code, ns)  # noqa: S102  (REPL 协议本体)
        except LevelUp:
            o = state["obs"]
            res.per_level_steps.append(game.steps - state["level_start"])
            state["level_start"] = game.steps
            outcome = f"\n🎉 过关! 现在 level {o.level}/{o.win_levels}" + (" 全部通关!" if o.done else "")
            log(f"  [{game.game_id}] level->{o.level} @ step {game.steps}")
        except BudgetExhausted:
            outcome = "\n(预算耗尽)"
        except _TimeoutError:
            outcome = f"\n(代码执行超过 {EXEC_TIMEOUT_S}s 被中断; 真机动作已生效不回滚)"
        except Exception:  # noqa: BLE001
            outcome = "\n执行异常:\n" + traceback.format_exc(limit=3)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)

        out = buf.getvalue()
        if len(out) > MAX_OUTPUT_CHARS:
            out = out[:MAX_OUTPUT_CHARS] + f"\n...(截断, 共{len(out)}字符)"
        status = (f"\n\n[round {rnd+1}/{max_rounds}] level {state['obs'].level}/{res.win_levels}, "
                  f"已用 {game.steps}/{max_actions} 动作。notes={str(ns['notes'])[:1200]}")
        exchanges.append(((out or "(无输出)") + outcome + status, raw[-2500:]))

    res.levels_completed = state["obs"].level
    res.steps = game.steps
    res.state = state["obs"].state
    res.seconds = round(time.monotonic() - t0, 1)
    return res
