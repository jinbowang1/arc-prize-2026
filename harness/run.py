"""主控循环与介入日志。

每关: probe -> percept -> 假设引擎 -> 兜底穷举 -> ask_human -> 回流再解 -> verify

人工只允许给"谓词族名 + 参数"; 每次 ask_human 计数一次, 介入次数就是这个
harness 的分数(越低越好)。介入日志落盘, 它同时是断网小模型要答的题库。

⚠️每关都要重新 probe: ls20/tr87/ft09 三局都是逐关升级机制, 上一关摸清的
交互规则在下一关可能整个作废。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import closedloop, hypo, model
from .env import Action, Game, Obs, action_space
from .hypo import GoalHypothesis, SubmitMatch, Transition
from .percept import analyze, discover_entities
from .probe import ProbeReport, run_probe
from .search import SearchResult, best_first, bfs_level_up


@dataclass
class LevelRecord:
    level: int
    solved: bool
    steps: int
    baseline: int | None
    seconds: float
    search: str
    intervened: bool = False

    @property
    def rhae(self) -> float | None:
        if not self.solved or not self.baseline or self.steps == 0:
            return None
        return min(1.15, (self.baseline / self.steps) ** 2)


@dataclass
class RunLog:
    game_id: str
    records: list[LevelRecord] = field(default_factory=list)
    interventions: list[dict] = field(default_factory=list)
    solution: list[list[str]] = field(default_factory=list)

    @property
    def intervention_count(self) -> int:
        return len(self.interventions)

    def summary(self) -> str:
        out = [f"=== {self.game_id} ==="]
        total_ai = total_hu = 0
        for r in self.records:
            mark = "✓" if r.solved else "✗"
            rh = f" RHAE={r.rhae*100:.1f}%" if r.rhae else ""
            iv = " [人工介入]" if r.intervened else ""
            out.append(f" {mark} L{r.level+1}: {r.steps} 步 (人类 {r.baseline}){rh}"
                       f" {r.seconds:.1f}s{iv}")
            if r.solved:
                total_ai += r.steps
                total_hu += r.baseline or 0
        n_done = sum(1 for r in self.records if r.solved)
        out.append(f" 通关 {n_done}/{len(self.records)} | AI {total_ai} 步 vs 人类 {total_hu} 步")
        out.append(f" **人工介入次数 = {self.intervention_count}**")
        return "\n".join(out)


def replay_frames(game: Game, base: Obs, seq: list[Action]) -> list[np.ndarray]:
    """在克隆体上重放一关的解, 收集**通关那一步之前**的逐帧网格。

    为什么止步于通关前一帧: 通关那一步之后引擎已经切到下一关, 那张图属于
    新关卡, 拿它当"目标达成的样子"会把两关的内容混在一起拟合。
    """
    c = game.fork()
    frames = [np.array(base.grid)]
    for a in seq[:-1]:
        o = c.act(a)
        frames.append(np.array(o.grid))
    return frames


def _evidence(h: GoalHypothesis, runs: list[list[np.ndarray]]) -> float:
    """证据强度 = 在已通关的解上, 这个量最多下降了多大比例(0~1)。

    "下降过"是个太松的判据 —— r11l L1 实测有 24 条假设全部通过, 靠族名
    字母序排队, color_count 就这么排到了第一。**下降幅度才是证据**:
    从 100 掉到 5 的量和从 2 掉到 0 的量, 不该同等对待。
    """
    best = 0.0
    for frames in runs:
        if len(frames) < 2:
            continue
        hs = [h.distance(f) for f in frames]
        # 999 是"定位不到"的哨兵值不是距离 —— (999-0)/999 会算出假"证据 100%"
        # (ls20 L2 实锤, 与 validate_heuristic 同一处坑)。定位不到的帧占多数
        # 时这关不算证据; 幅度只在可定位的值上算。
        loc = [x for x in hs if x < 999]
        if len(loc) * 2 < len(hs) or not loc or loc[0] <= 0:
            continue
        best = max(best, (loc[0] - min(loc)) / loc[0])
    return best


def learn_goals(samples: list[Transition], solved_runs: list[list[np.ndarray]]
                ) -> list[tuple[GoalHypothesis, str]]:
    """拟合目标假设, 再用已通关的解回放检验梯度。两道关缺一不可。

    fit() 只保证"这条假设在通关前成立、开局不成立", 那是**相关性**;
    validate_heuristic() 才回答"推着它走真的会靠近通关吗"。cd82 L3 花了
    四百多秒算力才买到这条教训 —— 有梯度 ≠ 梯度指向目标。

    排序: 关系型优先(唯一跨关泛化的), 然后按证据强度, 最后才是族名。
    """
    out = []
    for h in hypo.fit(samples):
        ok, note = hypo.validate_heuristic(h.distance, solved_runs)
        if ok:
            ev = _evidence(h, solved_runs)
            out.append((h, f"证据 {ev:.0%}; {note}", ev))
    out.sort(key=lambda x: (not hypo.is_relational(x[0]), -x[2]))
    return [(h, note + ("" if hypo.is_relational(h) else " ⚠️绝对型, 参数是上一关的常数"))
            for h, note, _ in out]


def _as_node_heuristic(h: GoalHypothesis, base_level: int):
    """把假设包成 best_first 要的 (node, obs) -> float。

    SubmitMatch 一族必须走 distance_on_node(它要 fork 出来 peek 一次提交
    动作), 其余族只看网格。这是这一族与其它族唯一的接口差别。
    """
    if isinstance(h, SubmitMatch):
        return lambda n, ob: h.distance_on_node(n, ob, base_level)
    return lambda n, ob: h.distance(np.array(ob.grid))


def solve_game(game_id: str, baselines: list[int] | None = None,
               max_nodes: int = 20000, max_seconds: float = 180.0,
               bfs_seconds: float = 60.0, goal=None, max_goals: int = 3,
               abstract_seconds: float = 30.0,
               ask_human=None, log_dir: str = "harness_runs") -> RunLog:
    """跑完一整个游戏。真机只走已验证的最短序列, 搜索全在克隆体上。

    每关通关都会回流成下一关的先验: 逐帧序列 -> 拟合目标假设 -> 梯度检验
    -> 下一关的最佳优先启发式。这是免费的监督信号, 通一关就多一个样本。
    """
    game, obs = Game.make(game_id)
    # 动作空间只能问引擎, 不能写死。之前这里是 [1,2,3,4,5,6], 到纯 click 的
    # r11l(available_actions=[6])上就会去试 5 个不存在的键。
    sp = action_space(list(obs.actions) or [1, 2, 3, 4, 5, 6])
    print(f"[run] {game_id} 动作空间={sp['kind']} keys={sp['keys']} clicks={sp['clicks']}", flush=True)
    log = RunLog(game_id=game_id)
    full_seq: list[Action] = []
    samples: list[Transition] = []            # 通关瞬间, 拟合目标用
    solved_runs: list[list[np.ndarray]] = []  # 已通关关卡的逐帧, 检验梯度用
    goals: list[tuple[GoalHypothesis, str]] = [(goal, "调用方指定")] if goal else []

    while obs.level < obs.win_levels:
        lv = obs.level
        t0 = time.time()

        scene = analyze(obs.grid)
        clicks = [Action.click(c, r) for (r, c) in scene.targets]
        rep = run_probe(game, obs, sp["kind"], sp["keys"], clicks)

        # 实体发现: 只问"画面上有几处互不相干的地方在动", 不问它有什么用。
        # 功能未知也要登记 —— cd82 L4 卡几小时的根因就是"探测不到功能就
        # 当它不存在", 把第二个面板整个丢了。
        ents = discover_entities(
            lambda a: np.array(game.peek(a).grid), np.array(obs.grid),
            [Action.key(i) for i in sp["keys"]] + clicks)
        print(f"L{lv+1} 实体 {len(ents)} 个: " +
              " | ".join(e.line() for e in ents[:4]), flush=True)

        budget = rep.depth_cap(100)
        click_id = 6 if sp["clicks"] else None

        # 两段式:先让 BFS 限时跑 —— 它解出来的就是**最短**序列, 而步数就是
        # 分数(RHAE 平方惩罚)。cd82 实测: 同一关 BFS 5 步 vs 启发式 12 步,
        # RHAE 115% vs 21%, 差 94 个百分点。所以能最优就别用近似。
        res = bfs_level_up(game, obs, sp["keys"], click_id, rep.mask,
                           max_depth=budget, max_nodes=max_nodes,
                           max_seconds=min(max_seconds, bfs_seconds))

        # 上一关学来的假设未必在这一关还成立, 先做开局体检再决定试不试。
        usable, rejected = [], []
        for h, note in goals:
            d0 = _as_node_heuristic(h, lv)(game.fork(), obs)
            if d0 >= 999:
                rejected.append(f"{h.describe()} — 本关定位不到目标对象")
            elif d0 == 0:
                # cd82 L3 的原样翻版: 起始 h 就是 0, 推它等于没推,
                # 最佳优先当场退化成广度优先, 白烧几百秒。
                rejected.append(f"{h.describe()} — 开局 h 已是 0, 该量与本关通关无关")
            else:
                usable.append((h, note, d0))
        if rejected:
            print(f"L{lv+1} 假设体检淘汰 {len(rejected)} 条: " +
                  "; ".join(rejected[:3]), flush=True)

        # BFS 交白卷才上启发式:它能到更深, 但给出的是近似解, 步数会差。
        # 假设逐条试 —— 一条假设错了只是慢, 不会给出错解, 因为过关判定
        # 始终只认引擎的 levels_completed。
        # ⚠️只试前 max_goals 条并显式报告丢掉了几条。静默截断会让"试过了都
        # 不行"和"根本没试"看起来一模一样。
        tried = usable[:max_goals]
        if len(usable) > len(tried):
            print(f"L{lv+1} ⚠️可用假设 {len(usable)} 条, 只试前 {len(tried)} 条, "
                  f"其余未试: " + "; ".join(h.describe() for h, _, _ in usable[len(tried):]),
                  flush=True)
        # 抽象层先上, 真机搜索垫后。理由是三个数量级的速度差:
        # 真机每扩展一个节点要 fork 一批克隆体(r11l 实测 8 次扩展/秒),
        # numpy 画布上是微秒级。抽象层出的只是**候选**, 必须回真机走一遍
        # 才算数 —— "搜索内通关 ≠ 解可复现"。
        models: dict[str, model.ActionModel] = {}
        if not res.solved and tried:
            acts_all = [Action.key(i) for i in sp["keys"]] + clicks
            models = model.learn(game, obs, acts_all, mask=rep.mask)
            print(f"L{lv+1} " + model.coverage(models), flush=True)

        for h, note, d0 in tried:
            if res.solved:
                break
            print(f"L{lv+1} 试假设 {h.describe()} [起始 h={d0:.0f}] [{note}]", flush=True)

            if not isinstance(h, SubmitMatch):
                p = model.plan(np.array(obs.grid), models, h.distance,
                               max_depth=min(budget, 12), max_seconds=abstract_seconds)
                print(f"L{lv+1}   {p.text()}", flush=True)
                if p.found:
                    # 闭环校验: 每步比对预测与实测, 第一次分歧就停。分歧不是
                    # 失败, 是"模型里少了什么"的指向 —— 而且它指向重采模型,
                    # 不是加大搜索。
                    loop = closedloop.run(game.fork(), obs, p.seq,
                                          model.predictor(models))
                    print(f"L{lv+1}   {loop.text()}", flush=True)
                    if loop.solved:
                        res = SearchResult(True, seq=loop.seq, nodes=p.nodes,
                                           seconds=p.seconds)
                        break

            res = best_first(game, obs, sp["keys"], click_id,
                             _as_node_heuristic(h, lv),
                             rep.mask, max_depth=budget, max_nodes=max_nodes * 10,
                             max_seconds=max_seconds)
            print(f"L{lv+1}   -> {res.text()}", flush=True)

        intervened = False
        if not res.solved:
            report = hypo.ask_human_report(
                rep.text(), scene.text(np.array(obs.grid)), res.text(),
                samples, [h for h, _ in goals])
            report += "\n[percept] 实体登记:\n" + "\n".join("    " + e.line() for e in ents)
            print(report, flush=True)
            if ask_human is not None:
                spec = ask_human(rep, scene, res, obs, report)
                log.interventions.append({
                    "level": lv, "probe": rep.text(), "search": res.text(),
                    "given": str(spec),
                })
                intervened = True
                # 谓词族回流后重搜的接口预留给 hypo 层, 当前版本直接判失败
                # (回流实现见 hypo.py; 未实现前不要假装它成功了)

        rec = LevelRecord(level=lv, solved=res.solved, steps=len(res.seq),
                          baseline=baselines[lv] if baselines and lv < len(baselines) else None,
                          seconds=time.time() - t0, search=res.text(),
                          intervened=intervened)
        log.records.append(rec)
        print(f"L{lv+1}: {res.text()}", flush=True)

        if not res.solved:
            break

        # 回流: 这一关的解就是下一关的监督信号
        frames = replay_frames(game, obs, res.seq)
        solved_runs.append(frames)
        if len(frames) >= 2:
            samples.append(Transition(before=frames[0], after=frames[-1], level=lv))
        goals = learn_goals(samples, solved_runs) or goals
        print(f"L{lv+1} 回流: 样本 {len(samples)} 个 -> 通过梯度检验的假设 "
              f"{len(goals)} 条" +
              ("; " + goals[0][0].describe() if goals else ""), flush=True)

        # 真机执行这一关的解
        for a in res.seq:
            obs = game.act(a)
        full_seq += res.seq
        log.solution.append([str(a) for a in res.seq])

    Path(log_dir).mkdir(exist_ok=True)
    stamp = f"{log_dir}/{game_id}.json"
    Path(stamp).write_text(json.dumps({
        "game": game_id,
        "levels": [r.__dict__ for r in log.records],
        "interventions": log.interventions,
        "solution": log.solution,
    }, ensure_ascii=False, indent=2))
    return log


def verify_replay(game_id: str, solution: list[list[str]]) -> tuple[bool, int, str]:
    """全新环境整条重放。唯一有效的判据 —— 搜索内通关不算数。

    分段解拼接的边界重复陷阱(ls20 L6 踩过): 每段快照可能已含穿关那一步,
    拼接时再补一次就会多走。整条重放能把这类错误全部暴露出来。
    """
    game, obs = Game.make(game_id)
    n = 0
    for seg in solution:
        for s in seg:
            a = _parse(s)
            obs = game.act(a)
            n += 1
            if obs.dead:
                return False, n, f"第 {n} 步 GAME_OVER"
    return obs.level >= obs.win_levels, n, f"通关 {obs.level}/{obs.win_levels} state={obs.state}"


def _parse(s: str) -> Action:
    """把 'A3' / 'A6(39,4)' 解析回 Action。"""
    if "(" not in s:
        return Action.key(int(s[1:]))
    head, rest = s.split("(", 1)
    x, y = rest.rstrip(")").split(",")
    return Action.click(int(x), int(y), int(head[1:]))
