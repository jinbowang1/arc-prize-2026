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

from .env import Action, Game, Obs, action_space
from .percept import analyze
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


def solve_game(game_id: str, baselines: list[int] | None = None,
               max_nodes: int = 20000, max_seconds: float = 180.0,
               bfs_seconds: float = 60.0, goal=None,
               ask_human=None, log_dir: str = "harness_runs") -> RunLog:
    """跑完一整个游戏。真机只走已验证的最短序列, 搜索全在克隆体上。"""
    game, obs = Game.make(game_id)
    sp = action_space([1, 2, 3, 4, 5, 6])
    log = RunLog(game_id=game_id)
    full_seq: list[Action] = []

    while obs.level < obs.win_levels:
        lv = obs.level
        t0 = time.time()

        scene = analyze(obs.grid)
        clicks = [Action.click(c, r) for (r, c) in scene.targets]
        rep = run_probe(game, obs, sp["kind"], sp["keys"], clicks)

        budget = rep.budget or 100
        click_id = 6 if sp["clicks"] else None

        # 两段式:先让 BFS 限时跑 —— 它解出来的就是**最短**序列, 而步数就是
        # 分数(RHAE 平方惩罚)。cd82 实测: 同一关 BFS 5 步 vs 启发式 12 步,
        # RHAE 115% vs 21%, 差 94 个百分点。所以能最优就别用近似。
        res = bfs_level_up(game, obs, sp["keys"], click_id, rep.mask,
                           max_depth=budget, max_nodes=max_nodes,
                           max_seconds=min(max_seconds, bfs_seconds))

        # BFS 交白卷才上启发式:它能到更深, 但给出的是近似解, 步数会差。
        if not res.solved and goal is not None:
            res = best_first(game, obs, sp["keys"], click_id,
                             lambda n, ob: goal.distance(np.array(ob.grid)),
                             rep.mask, max_depth=budget, max_nodes=max_nodes * 10,
                             max_seconds=max_seconds)

        intervened = False
        if not res.solved and ask_human is not None:
            spec = ask_human(rep, scene, res, obs)
            log.interventions.append({
                "level": lv,
                "probe": rep.text(),
                "percept": scene.text(np.array(obs.grid)),
                "search": res.text(),
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
