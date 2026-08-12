"""主循环:ReAct 与 Plan-and-Execute 的结合。

**为什么把这两个缝在一起(2026-08-12 用户定的方向):**

原来的主控是**纯开环、且每关从零开始**: probe → 穷举 → 假设 → 搜, 一次性
出完整方案, 执行到底才知道行不行。它有两个毛病:

1. **失败信号出现得极晚, 而且没有方向。** cd82 L4 的方案在抽象层"解出"了,
   真机走到第四笔才发现无解, 拿到的是一句"找不到"。
2. **上一关白学了。** 三局都是逐关升级机制, 上一关的知识大部分仍然成立,
   只加了一点新东西 —— 全扔掉重来是浪费, 全盘照搬又会中招(r11l 上从上一关
   抄来的常数型假设通过了全部检验, 白烧九分钟)。

所以每关的流程改成四段:

    ① 交接   和上一关做 diff: 新增了什么颜色/形状/动作? 继承来的知识
             逐条过体检, 过了才用, 没过要说明为什么 —— **复用但不全用**
    ② ReAct  只对「新增的东西」做定向探测: 这个新元素归哪个动作管?
             一次一个动作看结果。目的不是通关, 是把新元素的角色问清楚
    ③ Plan   用(继承 + 新学)的模型出全局最短计划。开环便宜, 而且它能给出
             全局最少步数 —— 步数就是分数(RHAE 平方惩罚)
    ④ 闭环执行  逐步比对预测与实测, **第一次分歧就停**。分歧不是失败, 是
             "模型里少了什么"的指向 —— 回 ② 定向补测, 回 ③ 重规划,
             🚨**而不是加大搜索**(cd82 那次一看到差 12 格就把 beam 从 40
             加到 800, 方向全错)

🚨**一个容易搞反的点: 闭环不是为了在真机上纠错。** 真机每一步都计分, 不能
拿它试错。整个 ①②③④ 循环全跑在克隆体上, 真机只走最后确认过的那一条。
闭环省的是**搜索预算**, 不是步数。

**交接单与闭环是互补的, 缺一不可。** 拿 cd82 实测: 交接单在 L2/L3 准确报出
新增颜色(那正是它逐关升级的方式), 但在 **L4 什么也报不出来** —— 而 L4 正是
我卡几小时的那关, 因为缺的第二个面板从 L1 就在, 它不是"新增"的。
**交接单只能告诉你这一关新加了什么, 告诉不了你你从头到尾就没看见什么。**
后者只有闭环的分歧抓得住。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import carryover, closedloop, factored, hypo, layers, model
from .carryover import Knowledge, LevelBrief
from .env import Action, Game, Obs, action_space
from .percept import analyze, click_targets, discover_entities
from .probe import run_probe
from .run import LevelRecord, RunLog, _as_node_heuristic, learn_goals, replay_frames
from .search import SearchResult, bfs_level_up, best_first


@dataclass
class LevelTrace:
    """一关里发生了什么。这是给人看的, 也是介入日志的正文。"""

    level: int
    brief: LevelBrief | None = None
    react: list[str] = field(default_factory=list)
    plans: list[str] = field(default_factory=list)
    divergences: list[str] = field(default_factory=list)
    solved_by: str = ""

    def text(self) -> str:
        out = []
        if self.brief:
            out.append(self.brief.text())
        for r in self.react:
            out.append("  [ReAct] " + r)
        for p in self.plans:
            out.append("  [Plan] " + p)
        for d in self.divergences:
            out.append("  [闭环] " + d)
        if self.solved_by:
            out.append(f"  ✅ 由「{self.solved_by}」解出")
        return "\n".join(out)


def react_probe(obs: Obs, brief: LevelBrief, ents: list) -> list[str]:
    """②ReAct: 对交接单点名的新元素, 问"它归谁管"。

    不问"这个动作有什么用"(那要等到能规划才知道), 只问"画面上这块新东西
    受哪些动作影响"。**功能未知也要登记** —— 用户玩 cd82 的原话:
    "一上来就看到了(小面板), 但不知道有什么用, 第二三关才意识到可以换
    油漆桶颜色"。登记实体和理解功能是两件事, 混为一谈就会把没搞懂用途的
    东西直接丢掉, 那正是 cd82 L4 卡几小时的根因。

    这一步是纯读取 —— 实体表是上游已经采好的, 这里只做归属匹配, 不额外
    花 fork。
    """
    g = np.array(obs.grid)
    out = []
    for c in brief.new_colors:
        pos = np.argwhere(g == c)
        if not len(pos):
            continue
        r0, r1 = int(pos[:, 0].min()), int(pos[:, 0].max())
        c0, c1 = int(pos[:, 1].min()), int(pos[:, 1].max())
        owners: list[str] = []
        for e in ents:
            er0, er1, ec0, ec1 = e.bbox
            if not (er1 < r0 or er0 > r1 or ec1 < c0 or ec0 > c1):
                owners += e.movers
        uniq = sorted(set(owners))
        if uniq:
            out.append(f"新增色{c} 所在区域受 {len(uniq)} 个动作影响: {uniq[:6]}")
        else:
            out.append(f"新增色{c} 所在区域**没有任何动作能改动** —— "
                       f"多半是题面/指示物, 不是可操作对象")
    if brief.new_shapes and not brief.new_colors:
        out.append(f"{brief.new_shapes} 种新形状但没有新颜色 —— "
                   f"变的是排布不是元素")
    return out


def resample_after_divergence(game: Game, obs: Obs, div, models: dict,
                              mask) -> str:
    """④→② 的回流: 分歧发生在哪一步, 就在**那个状态**重采那个动作。

    这是 CEGIS 的老套路(ls20 L5 用过, cd82 从头到尾没用, 是退步)。要点是
    重采必须在**分歧现场的状态**上做, 而不是回到关卡开局重采一遍 —— 分歧
    恰恰说明这个动作的效果与状态有关, 在开局采一百次也采不出来。
    """
    fresh = model.learn(game, obs, [div.action], n_states=3, mask=mask)
    m = fresh.get(repr(div.action))
    if m is None:
        return f"{div.action} 重采失败"
    models[repr(div.action)] = m
    return f"在分歧现场重采 {div.action}: {m.line()}"


def solve_level(game: Game, obs: Obs, know: Knowledge, prev_scene,
                sp: dict, cfg: dict) -> tuple[SearchResult, LevelTrace, dict]:
    """跑完一关。返回 (结果, 过程记录, 这一关学到的东西)。"""
    lv = obs.level
    tr = LevelTrace(level=lv)
    killed: set[str] = set()
    scene = analyze(obs.grid)
    clicks = [Action.click(c, r) for (r, c) in scene.targets]
    acts = [Action.key(i) for i in sp["keys"]] + clicks
    click_id = 6 if sp["clicks"] else None

    # ① 交接
    tr.brief = carryover.brief(lv, obs, scene, know if know.learned_at >= 0 else None,
                               prev_scene)

    rep = run_probe(game, obs, sp["kind"], sp["keys"], clicks)
    ents = discover_entities(lambda a: np.array(game.peek(a).grid),
                             np.array(obs.grid), acts)
    budget = rep.depth_cap(100)

    # ② ReAct: 只探新东西
    tr.react = react_probe(obs, tr.brief, ents)
    tr.react.append(f"实体 {len(ents)} 个, 动作预算 {rep.budget}({rep.budget_note})")

    # 继承的目标假设过体检 —— 复用但不全用
    keep, drop = carryover.vet_goals(know, obs, lambda h: _as_node_heuristic(h, lv),
                                     game.fork)
    tr.brief.inherited += [f"目标假设 {h.describe()}(h={d:.0f})" for h, _, d in keep[:3]]
    tr.brief.dropped += drop[:3]

    # ③ Plan —— 由便宜到贵, 由最短到近似
    def cands(o: Obs) -> list[Action]:
        return ([Action.key(i) for i in sp["keys"]] +
                [Action.click(c, r, 6) for (r, c) in click_targets(np.array(o.grid))])

    # 3a) BFS: 解出来的就是最短序列, 而步数就是分数。能最优就别用近似。
    res = bfs_level_up(game, obs, sp["keys"], click_id, rep.mask,
                       max_depth=budget, max_nodes=cfg["max_nodes"],
                       max_seconds=cfg["bfs_seconds"])
    tr.plans.append("BFS(最短) " + res.text())
    if res.solved:
        tr.solved_by = "BFS 最短路"
        return res, tr, {"scene": scene, "slots": know.slots, "killed": killed}

    # 3b) 槽结构: 继承的先过体检, 没有或不合格就重新分
    live = [p.action for p in rep.profiles if not p.is_noop and not p.kills]
    sm, why = carryover.vet_slots(know, live)
    if sm is None and len(live) >= 4:
        sm = factored.learn_slots(game, obs, acts, rep.mask)
        why = "本关重新分槽"
    if sm is not None:
        tr.plans.append(f"{why} -> {sm.text()}")

        # 3b-1) 分层渲染模型: 抽象层排序 + 真机判定。
        # 模型不准也能用 —— 它只决定先试哪个候选, 过关只认引擎。
        lm = layers.learn_layers(game, obs, sm, rep.mask)
        tr.plans.append(lm.text())
        if lm.usable_for_ranking:
            for h, note, d0 in list(keep[:cfg["max_goals"]]):
                lp = layers.plan_and_verify(game, obs, lm, sm, h.distance,
                                            mask=rep.mask, top_k=cfg["top_k"],
                                            max_seconds=cfg["layer_seconds"])
                tr.plans.append(f"分层排序+真机验({h.describe()}) " + lp.text())
                if lp.found:
                    tr.solved_by = f"分层排序+真机验({h.describe()})"
                    return SearchResult(True, seq=lp.seq, seconds=lp.seconds), tr, \
                        {"scene": scene, "slots": sm, "killed": killed}
                if lp.disproved:
                    # 反例回流: 拉黑这条假设, 本关不再试, 也不带去下一关。
                    # 硬否证比统计证据强 —— 别让一条已经被真机打死的假设
                    # 继续排在候选第一位。
                    killed.add(h.describe())
                    keep = [k for k in keep if k[0].describe() != h.describe()]
                    tr.brief.dropped.append(f"{h.describe()} — 🚨真机把 h 推到 0 仍未过关, 已证伪")

        # 3b-2) 槽搜索: 允许多于"每槽一次"的组合(新动作不受约束)
        sr = factored.slot_search(game, obs, sm, mask=rep.mask, candidates=cands,
                                  max_seconds=cfg["slot_seconds"])
        tr.plans.append(sr.text())
        if sr.solved:
            tr.solved_by = "槽搜索"
            return SearchResult(True, seq=sr.seq, seconds=sr.seconds), tr, \
                {"scene": scene, "slots": sm, "killed": killed}

    # 3c) 抽象模型 + ④闭环执行, 分歧就回流重采再规划
    models = model.learn(game, obs, acts, mask=rep.mask)
    tr.plans.append(model.coverage(models))
    for h, note, d0 in keep[:cfg["max_goals"]]:
        for rnd in range(cfg["loop_rounds"]):
            p = model.plan(np.array(obs.grid), models, h.distance,
                           max_depth=min(budget, 12), max_seconds=cfg["abstract_seconds"])
            tr.plans.append(f"{h.describe()} 第{rnd+1}轮 {p.text()}")
            if not p.found:
                break
            loop = closedloop.run(game.fork(), obs, p.seq, model.predictor(models))
            tr.divergences.append(loop.text())
            if loop.solved:
                tr.solved_by = f"抽象规划+闭环({h.describe()})"
                return SearchResult(True, seq=loop.seq), tr, {"scene": scene, "slots": sm, "killed": killed}
            if loop.divergence is None:
                break        # 计划走完没通关, 是目标假设不对, 换一条假设
            # ④→②: 在分歧现场重采, 然后重规划。**不是加大搜索。**
            here = game.fork()
            for a in loop.seq[:-1]:
                here.act(a)
            cur = here.act(loop.seq[-1]) if loop.seq else obs
            tr.react.append(resample_after_divergence(here, cur, loop.divergence,
                                                      models, rep.mask))

    # 3d) 兜底: 真机最佳优先
    for h, note, d0 in keep[:cfg["max_goals"]]:
        res = best_first(game, obs, sp["keys"], click_id, _as_node_heuristic(h, lv),
                         rep.mask, max_depth=budget, max_nodes=cfg["max_nodes"] * 10,
                         max_seconds=cfg["best_first_seconds"])
        tr.plans.append(f"真机最佳优先({h.describe()}) " + res.text())
        if res.solved:
            tr.solved_by = "真机最佳优先"
            break

    return res, tr, {"scene": scene, "slots": sm, "killed": killed}


DEFAULT_CFG = {
    "max_nodes": 20000,
    "bfs_seconds": 45.0,
    "slot_seconds": 120.0,
    "abstract_seconds": 30.0,
    "best_first_seconds": 120.0,
    "max_goals": 2,
    "loop_rounds": 3,
    "top_k": 60,
    "layer_seconds": 60.0,
}


def solve_game(game_id: str, baselines: list[int] | None = None,
               cfg: dict | None = None) -> RunLog:
    """跨关主循环。知识在 Knowledge 里跨关携带, 每关入口处过体检。"""
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    game, obs = Game.make(game_id)
    sp = action_space(list(obs.actions) or [1, 2, 3, 4, 5, 6])
    print(f"[agent] {game_id} 动作空间={sp['kind']}", flush=True)

    log = RunLog(game_id=game_id)
    know = Knowledge()
    prev_scene = None
    samples: list[hypo.Transition] = []
    solved_runs: list[list[np.ndarray]] = []

    while obs.level < obs.win_levels:
        lv = obs.level
        t0 = time.time()
        res, tr, learned = solve_level(game, obs, know, prev_scene, sp, cfg)
        print(tr.text(), flush=True)

        log.records.append(LevelRecord(
            level=lv, solved=res.solved, steps=len(res.seq),
            baseline=baselines[lv] if baselines and lv < len(baselines) else None,
            seconds=time.time() - t0, search=tr.solved_by or "未解出"))
        if not res.solved:
            break

        # 沉淀: 这一关的解就是下一关的监督信号
        frames = replay_frames(game, obs, res.seq)
        solved_runs.append(frames)
        if len(frames) >= 2:
            samples.append(hypo.Transition(before=frames[0], after=frames[-1], level=lv))
        fresh = learn_goals(samples, solved_runs) or know.goals
        dead = learned.get("killed") or set()
        know.goals = [(h, n) for h, n in fresh if h.describe() not in dead]
        if dead:
            print(f"  已证伪并拉黑 {len(dead)} 条假设: {sorted(dead)}", flush=True)
        know.slots = learned.get("slots")
        know.colors |= set(int(x) for x in np.unique(np.array(obs.grid)))
        know.shapes |= carryover.shape_keys(np.array(obs.grid))
        know.learned_at = lv
        prev_scene = learned.get("scene")
        print(f"  沉淀: 目标假设 {len(know.goals)} 条, 见过颜色 {len(know.colors)} 种, "
              f"形状等价类 {len(know.shapes)} 种", flush=True)

        for a in res.seq:
            obs = game.act(a)
        log.solution.append([str(a) for a in res.seq])

    return log
