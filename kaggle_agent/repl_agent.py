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

# 全貌注入(08-23 用户定的方向): 模型此前连自己在打比赛都不知道,
# 预算决策("过第1关最值钱/别恋战")没法做。
META = """【你在打比赛】约上百个游戏流水作业, 每个游戏只分到几分钟和有限动作数。
计分: 每过一关按 (人类步数/你的步数)² 得分, 只有过掉的关才得分; 没过关时
花掉的步数一分不扣——所以宁可多花步数也要过关, 千万别为省步数错过关。
第 1 关是教学关, 最容易也最值钱。一条路久攻不下就果断换思路, 恋战就是亏。"""

# 种类手册上场版(源头=notes/game-genre-field-guide.md, 八游戏一手攻关蒸馏;
# 改这里要同步改那边)。
FIELD_GUIDE = """【背景知识】你玩的是人类设计的小型解谜游戏: 一定能通关、不会太难。
老玩家经验如下, 先对号入座判断种类再按打法行动; 种类只是猜测, 以实际反馈为准。

通用规律(前人实测):
- 目标几乎总在画面上明示: 动作改得动的区域是你的"答案区", 改不动却有内容
  的区域是"题面"。过关常常就是把答案区弄成题面的样子。
- 有些格子每步自己变(步数条/计时器), 与你的动作无关, 别当机制。
- 效果和过关信号都可能滞后一拍: "按了没反应"先再按一下看。
- 点击的有效目标是物体(连通块), 别扫坐标。
- 图形匹配一般不看朝向(旋转镜像不算), 但颜色算。
- 单个动作全都"无效"时, 试两两组合(可能要"选中+放置"成对用)。

常见种类与打法:
A 复制图案: 题面+答案区并排 → 弄清"笔"是什么(大笔/小笔/换色/选零件+放置),
  从题面反推笔画; 注意后涂盖先涂。
B 导航开锁: 方向键控制一个角色, 锁上显示要求 → 踩机关把自己变成要求的形状
  颜色, 绕开会破坏状态的格子, 走进锁; 注意能量条。
C 推箱子: 可动块+插槽 → 只能推不能拉, 别推进死角。
D 点灯开关: 点一格翻一片 → 先摸清翻转模式, 想好该点的集合再动手。
E 词典翻译: 成对样例+待填区 → 从样例归纳映射, 循环键换候选填进去, 全对自动
  过关; 匹配不看朝向。
F 画符施法: 小画布+图案提示+角色+出口 → 画对图案是放技能, 过关要角色走进
  出口, 画布本身不是目标。
G 连锁消除: 点关键位置引发大连锁 → 优先点中心/缺口/对称点。
都不像: 找答案区和题面, 每个动作试一次记效果, 把"与题面的差距"做小。"""

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
- probe_keys(): 每个可用按键各按一次(真的花预算!), 返回每键改动区域的摘要 ——
  开局想快速摸清按键效果时用它, 比自己逐个试省代码。
- remember(text): 记进跨游戏记事本(下个游戏开局能看到); 过关后把"这游戏是
  什么种类、怎么过的、有什么坑"记一条, 后面的游戏靠它加速。
- save_tool(name, source): 把一个通用函数的源码存入跨游戏工具库(下个游戏开局
  自动加载); 只存验证过、跟具体游戏无关的工具。

工作流程(按顺序):
0. 对号入座: 对照下方"背景知识"判断这是哪类游戏(说出证据), 按该类打法行动;
1. 探索: 先用**少量**实验(几个 act 或一次 probe_keys)摸动作效果, 用 patch/diff
   看清每次改了什么; 看不懂再逐步加码 —— 实验是手段不是任务, 随时准备直接去赢;
2. 验证: 把猜出的规则写成 predict 函数, 用 verify_wm 验证 —— **没有通过 verify_wm 的规则不许用来规划**(经验: 不验证的世界模型比没有世界模型还糟);
3. 简化: 规则验证通过后, 试着把它改写得更简单(合并分支/去掉特例), 再验一次 —— 更简单且同样全对的规则泛化更好;
4. 规划: 在验证过的规则上写代码推演出过关序列, 再用最少的 act 执行;
5. 复杂关卡可以只对"关键子系统"建模(比如只预测被点击区域), 但验证不能省。
你在代码里定义的函数和变量会跨轮保留 —— 把好用的分析工具、验证过的 predict 沉淀下来复用。
print 出你想看的东西; 输出会截断到 3000 字符, 别打印整个 grid(用 patch 看局部)。

""" + META + "\n\n" + FIELD_GUIDE


class BudgetExhausted(Exception):
    pass


class LevelUp(Exception):
    pass


def _extract_code(text: str) -> str | None:
    """抠出要执行的代码块。

    🚨Qwen3.8 实测会把画面数据用裸 ``` 引用块贴回来(贴的是数据不是代码),
    "见围栏就执行"会把 `500f005` 这种像素行喂给 exec → SyntaxError 白烧一轮。
    判据改为**编译通过才算代码**: 带 python 标签的优先, 其次裸块, 都必须
    compile() 得过; 多个合法块取最后一个(模型习惯先探讨后给最终版)。
    """
    if "```" not in text:
        return None
    tagged, plain = [], []
    for part in text.split("```")[1::2]:  # 奇数段=围栏内
        body = part
        is_tagged = False
        if body.startswith("python"):
            body, is_tagged = body[6:], True
        elif body.startswith("py\n"):
            body, is_tagged = body[3:], True
        body = body.strip("\n")
        if not body.strip():
            continue
        try:
            compile(body, "<candidate>", "exec")
        except SyntaxError:
            continue
        (tagged if is_tagged else plain).append(body)
    if tagged:
        return tagged[-1]
    if plain:
        return plain[-1]
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


def _full_frame(g):
    """整幅画面(行号在左, 每格一个十六进制色号)。

    彩排实锤(08-22, gpu_v3_out/transcript): 不给全图, Qwen3.8 就自己烧 2-3 轮
    打印它 —— 27B 一轮 ~64s, 真赛场每局只摊到 4-6 轮, 观察轮必须由我们代替。
    ~4.3KB(≈1500 token) 换回 2-3 轮, 稳赚。
    """
    ruler = "    " + "".join(str(x % 10) for x in range(len(g[0])))
    rows = [f"{y:2d}  " + "".join(format(c, "x") if 0 <= c < 16 else "?" for c in row)
            for y, row in enumerate(g)]
    return ruler + "\n" + "\n".join(rows)


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
    transcript_path: str | None = None,
    home: str | None = None,
) -> GameResult:
    """transcript_path: 每轮(模型输出, 执行结果)落 jsonl —— 今晚两次都是靠
    transcript 破的案(空回复伪装成模型不行/提示词回归), 评测不许裸跑。

    home: 跨游戏持久目录(记事本 notes.md + 工具库 tools/*.py)。110 个隐藏游戏
    是流水打的, 打到后面, 记事本和工具库就是模型自己写的攻略——这是把
    "记忆复盘/工具积累"(手工攻关赢下六局的两大支柱)塞进沙箱的最小做法。"""
    import json as _json
    from pathlib import Path

    def _rec(**fields) -> None:
        """对话记录逐轮落盘。08-22 用户定的规矩: 模型看到的(提示词/喂回)和说出的
        (完整输出)都要记, 否则没法区分"提示词问题"还是"agent 问题"。"""
        if transcript_path:
            with open(transcript_path, "a") as f:
                f.write(_json.dumps(fields, ensure_ascii=False) + "\n")
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

    def probe_keys():
        """每个可用按键各按一次(真机计分!), 打印并返回每键改动摘要。"""
        keys = [a for a in (state["obs"].actions or (1, 2, 3, 4, 5)) if int(a) != 6]
        lines = []
        for k in keys:
            before = state["obs"].grid
            r = act(k)
            if r == -1:
                lines.append(f"A{k}: 导致死亡(已重置回本关起点)")
                continue
            d = _diff(before, state["obs"].grid)
            if not d:
                lines.append(f"A{k}: 无变化")
            else:
                ys = [c[0] for c in d]; xs = [c[1] for c in d]
                pair = {}
                for _, _, o_, n_ in d:
                    pair[(o_, n_)] = pair.get((o_, n_), 0) + 1
                top = sorted(pair.items(), key=lambda t: -t[1])[:3]
                chg = " ".join(f"{o_}→{n_}×{c}" for (o_, n_), c in top)
                lines.append(f"A{k}: 改{len(d)}格 行{min(ys)}-{max(ys)}列{min(xs)}-{max(xs)} 主要{chg}")
        out = "\n".join(lines)
        print(out)
        return out

    home_dir = Path(home) if home else None
    tools_dir = home_dir / "tools" if home_dir else None
    if tools_dir:
        tools_dir.mkdir(parents=True, exist_ok=True)

    def remember(text):
        """记进跨游戏记事本。"""
        if home_dir:
            with open(home_dir / "notes.md", "a") as f:
                f.write(f"[{game.game_id}] {str(text).strip()}\n")
        print("已记入记事本")

    def save_tool(name, source):
        """通用函数源码入库, 下个游戏开局自动加载。先编译再执行, 坏代码不入库。"""
        code = compile(source, f"<tool:{name}>", "exec")
        exec(code, ns)  # noqa: S102
        if tools_dir:
            (tools_dir / f"{name}.py").write_text(source)
        print(f"工具 {name} 已保存")

    ns: dict = {
        "act": act, "history": [], "components": _components, "diff": _diff,
        "patch": _patch, "notes": {}, "verify_wm": verify_wm,
        "probe_keys": probe_keys, "remember": remember, "save_tool": save_tool,
    }

    # 装载前面游戏留下的工具与笔记
    loaded_tools: list[str] = []
    if tools_dir:
        for f in sorted(tools_dir.glob("*.py")):
            try:
                exec(compile(f.read_text(), str(f), "exec"), ns)  # noqa: S102
                loaded_tools.append(f.stem)
            except Exception:  # noqa: BLE001
                pass  # 坏工具跳过, 不拖垮开局
    past_notes = ""
    if home_dir and (home_dir / "notes.md").exists():
        past_notes = (home_dir / "notes.md").read_text()[-2000:]

    def _sync():
        o = state["obs"]
        ns.update(grid=o.grid, level=o.level, win_levels=o.win_levels,
                  steps_used=game.steps, steps_budget=max_actions)

    _sync()
    # exchanges 存 (喂回的user消息, 模型的assistant输出); 组装时保持
    # system, first_user, a1, u1, a2, u2... 的严格交替 —— 最新结果只出现一次
    exchanges: deque[tuple[str, str]] = deque(maxlen=KEEP_EXCHANGES)
    first_user = (f"开局: level {obs.level}/{obs.win_levels}, 预算 {max_actions} 动作, "
                  f"可用动作 {list(obs.actions or (1,2,3,4,5,6))}\n\n"
                  f"完整画面:\n{_full_frame(obs.grid)}\n\n{_grid_summary(obs.grid)}"
                  + (f"\n\n你的跨游戏记事本(此前游戏记下的):\n{past_notes}" if past_notes else "")
                  + (f"\n\n工具库已加载: {loaded_tools}" if loaded_tools else ""))
    _rec(system=SYSTEM_PROMPT, opening=first_user)

    # 复读断路器: 关思考+低温下 Qwen3 会逐字复读上一轮(08-23 彩排 r11l 23 轮
    # 全灭实锤)。重复的纯观察代码不再执行, 注入干预并临时提温打散循环。
    last_code: str | None = None
    boost_temp = False
    for rnd in range(max_rounds):
        if time.monotonic() >= deadline or game.steps >= max_actions or state["obs"].done:
            break
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": first_user}]
        for u, a in exchanges:
            msgs += [{"role": "assistant", "content": a}, {"role": "user", "content": u}]
        try:
            raw = llm.chat(msgs, **({"temperature": 1.0} if boost_temp else {}))
        except Exception as e:  # noqa: BLE001
            log(f"  [{game.game_id}] LLM 失效: {e!r}")
            break
        code = _extract_code(raw)
        if code and code == last_code and "act(" not in code:
            boost_temp = True
            status = (f"[round {rnd+1}/{max_rounds}] level {state['obs'].level}/{res.win_levels}, "
                      f"已用 {game.steps}/{max_actions} 动作")
            fed = ("⚠️ 这段代码和上一轮完全相同, 已执行过, 不再重复执行。"
                   "画面没有变化, 光看是看不出新信息的; 本轮必须写不同的代码, "
                   "并至少包含一个此前没试过的 act() 动作。\n" + status)
            exchanges.append((fed, raw[-1500:]))
            _rec(round=rnd + 1, assistant=raw, result="(重复代码, 未执行)", fed_back=fed)
            continue
        if code:
            boost_temp = False
            last_code = code
        if not code:
            status = (f"[round {rnd+1}/{max_rounds}] level {state['obs'].level}/{res.win_levels}, "
                      f"已用 {game.steps}/{max_actions} 动作")
            fed = "(没有找到 ```python 代码块。请只输出一个代码块。)\n" + status
            exchanges.append((fed, raw[-1500:]))
            _rec(round=rnd + 1, assistant=raw, result="", fed_back=fed)
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
            outcome = (f"\n🎉 过关! 现在 level {o.level}/{o.win_levels}" + (" 全部通关!" if o.done else "")
                       + ("" if o.done else f"\n新关卡完整画面:\n{_full_frame(o.grid)}"))
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
        fed = (out or "(无输出)") + outcome + status
        exchanges.append((fed, raw[-2500:]))
        _rec(round=rnd + 1, assistant=raw, result=(out or "") + outcome, fed_back=fed)

    res.levels_completed = state["obs"].level
    res.steps = game.steps
    res.state = state["obs"].state
    res.seconds = round(time.monotonic() - t0, 1)
    return res
