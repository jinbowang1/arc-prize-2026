"""绘制类游戏的通用求解:自动认出答案区与提交动作, 然后走画笔库那条路。

`paint.py` 当年解开了 cd82, 但它有两个洞是手工传参的:
    answer_box  答案区在哪
    submit      哪个动作是"提交"
这个模块把这两个洞堵上, 于是那套方法变成通用的。

**堵法都来自因果结构, 不来自像素:**

  - **答案区** = 动作能改、且改得动的那块(hypo.propose_prompt_answer 已经能
    自动找, cd82 上首选提议与地面真值逐格相同)。
  - **提交动作** = 会改动答案区的动作; **调整动作** = 改不动答案区的动作。
    这个二分是这类游戏的骨架, 而不是 cd82 的特例。

**为什么这个骨架成立(paint.execute 里那句话):**
"提交动作不改变画笔构型, 所以第 i+1 笔是从第 i 笔的构型接着调。"
即: 构型(面板位置/当前颜色…)只被调整动作改, 提交不碰它; 画布只被提交改,
按当前构型盖。两层互不干扰 —— 所以画笔库采一次就一直有效,
不会"盖完第一笔构型就变了"。

**双底采集也能自动化。** cd82 当年是手工把答案区铺满两种底色。通用做法:
`floor_b = floor_a 上先按一次任意提交` —— 画布变了, **构型没变**,
正好满足"两底走同一条调整序列必到同一构型"的前提, 不需要任何游戏特定手段。
双底是必须的: 单底采到的笔看不见"涂成透明"的格子, cd82 上一笔曾被记成覆盖
12 格, 真身是覆盖 50+ 格擦掉 38 格, 据此排的方案真机必然无解。
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .env import Action, Game, Obs
from .search import fingerprint


def _region(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, r1, c0, c1 = box
    return np.array(grid[r0:r1 + 1, c0:c1 + 1])


def _config_fp(grid: np.ndarray, box: tuple[int, int, int, int],
               mask: np.ndarray | None) -> bytes:
    """构型指纹 = **把答案区挖掉之后**的画面。

    画布本身不属于构型 —— 不挖掉它, 同一个构型下涂了不同内容的两个状态会被
    当成两个构型, 画笔库瞬间爆炸而且全是重复的。
    """
    g = grid.copy()
    r0, r1, c0, c1 = box
    g[r0:r1 + 1, c0:c1 + 1] = 0
    return (g * mask).tobytes() if mask is not None else g.tobytes()


@dataclass
class Brush:
    """一支笔: 走 seq 调整到某构型, 再按 submit, 会在答案区盖出 stroke。"""

    seq: list[Action]
    submit: Action
    covered: np.ndarray          # 布尔, 哪些格真的被这一笔盖到
    stroke: np.ndarray           # 盖成什么色(只在 covered 上有意义)

    @property
    def size(self) -> int:
        return int(self.covered.sum())

    def apply(self, canvas: np.ndarray) -> np.ndarray:
        out = canvas.copy()
        out[self.covered] = self.stroke[self.covered]
        return out


@dataclass
class CanvasSetup:
    answer_box: tuple[int, int, int, int]
    submitters: list[Action] = field(default_factory=list)
    adjusters: list[Action] = field(default_factory=list)

    def text(self) -> str:
        return (f"[canvas] 答案区 {self.answer_box} | 提交动作 {len(self.submitters)} 个 "
                f"{[repr(a) for a in self.submitters[:4]]} | 调整动作 {len(self.adjusters)} 个")


def classify(game: Game, obs: Obs, acts: list[Action],
             answer_box: tuple[int, int, int, int]) -> CanvasSetup:
    """按"改不改得动答案区"把动作二分。"""
    st = CanvasSetup(answer_box=answer_box)
    before = _region(np.array(obs.grid), answer_box)
    for a in acts:
        o = game.peek(a)
        if o.dead:
            continue
        after = _region(np.array(o.grid), answer_box)
        (st.submitters if not np.array_equal(before, after) else st.adjusters).append(a)
    return st


def _inverse_pairs(game: Game, obs: Obs, adjusters: list[Action], base_cfg: bytes,
                   box: tuple[int, int, int, int], mask, limit: int = 8
                   ) -> list[tuple[Action, Action]]:
    """找"走 a 再走 b 构型回原处"的调整动作对。判据只看构型, 不看画布。"""
    out = []
    for a in adjusters[:limit]:
        ca = game.fork()
        if ca.act(a).dead:
            continue
        for b in adjusters[:limit]:
            if repr(a) == repr(b):
                continue
            cb = ca.fork()
            o = cb.act(b)
            if not o.dead and _config_fp(np.array(o.grid), box, mask) == base_cfg:
                out.append((a, b))
                break
    return out


def collect_brushes(game: Game, obs: Obs, st: CanvasSetup,
                    mask: np.ndarray | None = None,
                    max_configs: int = 400, max_seconds: float = 180.0
                    ) -> tuple[list[Brush], bool, int, int, int]:
    """枚举可达构型 × 提交动作, 双底采出每支笔的真实覆盖区。

    返回 (画笔库, 构型是否枚举完)。
    🚨**第二个返回值必须往上报。** 库不全时抽象层会稳定收敛到一个非零差异,
    看起来像"这关无解" —— 那是采集被截断, 不是游戏无解。
    """
    t0 = time.time()
    box = st.answer_box

    # 多个底。构型都不变(提交不改构型), 只有画布内容不同。
    #
    # 🚨**判据的前提是"两底在这一格上本来就不同"。** 第一版只用两个底
    # (b = a 上按一次提交), 那一笔只改了几十格, 剩下的格子两底本来就一样,
    # 于是"结果也一样"被当成了"这一笔盖过" —— 报出**一支盖满整个 10×10
    # 答案区的笔**, 全是假的。当年手工铺两种纯色底是处处不同的, 自动版
    # 把这个前提丢了。
    #
    # 所以: 多铺几个底, 并且**只在"底之间确实不同"的格子上做判定**;
    # 底都一样的格子这次采集没有证据, 既不能算覆盖, 也不能装作知道。
    base_cfg = _config_fp(np.array(obs.grid), box, mask)
    floors: list[tuple[Game, np.ndarray]] = [(game.fork(), _region(np.array(obs.grid), box))]

    def add_floor(f: Game, o: Obs) -> bool:
        """只收**构型没变**的底。构型变了的底走同一条调整序列会到别处,
        双底判据的前提就没了。这里不靠假设, 走完直接比对构型指纹自证。"""
        if o.dead or o.level != obs.level:
            return False
        if _config_fp(np.array(o.grid), box, mask) != base_cfg:
            return False
        floors.append((f, _region(np.array(o.grid), box)))
        return True

    for sub in st.submitters[:4]:
        f = game.fork()
        add_floor(f, f.act(sub))

    # 🚨从当前构型再提交往往是**幂等**的(刚涂完的东西再涂一次没变化),
    # 于是所有底长得一模一样, 能判的格子降到 0 —— cd82 L3 实测: 第 1 笔
    # 能判 50/100 格, 第 2 笔起判 **0/100**, 画笔覆盖被系统性低估, 最后
    # 卡在差 12 格说"没有任何一笔能缩小差异"。
    #
    # 造出真正不同的底要**去别的构型涂一笔再走回来**: 画布变了, 构型回原处。
    # 走回来靠 probe 已经测出来的互逆动作对; 而且不靠假设 —— 回来之后比对
    # 构型指纹, 对不上就不收这个底。
    for a, b in _inverse_pairs(game, obs, st.adjusters, base_cfg, box, mask)[:3]:
        for sub in st.submitters[:2]:
            f = game.fork()
            o = f.act(a)
            if o.dead:
                continue
            o = f.act(sub)
            if o.dead:
                continue
            o = f.act(b)
            if add_floor(f, o):
                break

    seen = {_config_fp(np.array(obs.grid), box, mask)}
    q: deque[tuple[list[Action], Game, Obs]] = deque([([], game.fork(), obs)])
    configs: list[list[Action]] = [[]]
    while q and len(configs) < max_configs and time.time() - t0 < max_seconds:
        seq, node, ob = q.popleft()
        for a in st.adjusters:
            child = node.fork()
            o = child.act(a)
            if o.dead or o.level > obs.level:
                continue
            fp = _config_fp(np.array(o.grid), box, mask)
            if fp in seen:
                continue
            seen.add(fp)
            configs.append(seq + [a])
            q.append((seq + [a], child, o))
    complete = not q

    # 哪些格子这次采集**有证据可判**: 底之间确实不同的格子。
    # 底都一样的格子, 这次采集给不出任何信息 —— 报出来的数字必须说清楚
    # 判了几格, 否则"覆盖 100 格"这种假消息会一路传到抽象层。
    bases = [reg for _, reg in floors]
    testable = np.zeros_like(bases[0], dtype=bool)
    for i in range(len(bases)):
        for j in range(i + 1, len(bases)):
            testable |= (bases[i] != bases[j])

    out: dict[tuple[bytes, bytes], Brush] = {}
    for seq in configs:
        if time.time() - t0 > max_seconds * 2:
            complete = False
            break
        clones = [f.fork() for f, _ in floors]
        bad = False
        for a in seq:
            for c in clones:
                if c.act(a).dead:
                    bad = True
            if bad:
                break
        if bad:
            continue
        for sub in st.submitters:
            regs = [_region(np.array(c.fork().act(sub).grid), box) for c in clones]
            same = np.ones_like(regs[0], dtype=bool)
            for r in regs[1:]:
                same &= (regs[0] == r)
            # 两条独立的证据, 取并集:
            #  ① 两底不同而结果相同 -> 这一格被这笔写死了(经典双底判据)
            #  ② 结果与某个底原本的值不同 -> 它当然被改过, 不需要两底不同
            # 只用①会把"底本来就一样"的格子全判成无证据: cd82 实测能判的
            # 格子只有 50/100, 而抽象层的地板恰好就是差 50 格 —— 一格不差,
            # 说明地板完全是采集判不了的那半边造成的, 不是搜索不行。
            changed = np.zeros_like(same, dtype=bool)
            for base_reg in bases:
                changed |= (regs[0] != base_reg)
            covered = same & (testable | changed)
            if not covered.any():
                continue
            stroke = np.zeros_like(regs[0])
            stroke[covered] = regs[0][covered]
            key = (covered.tobytes(), stroke.tobytes())
            if key not in out or len(seq) < len(out[key].seq):
                out[key] = Brush(seq=list(seq), submit=sub,
                                 covered=covered, stroke=stroke)
    return list(out.values()), complete, int(testable.sum()), int(testable.size), len(configs)


@dataclass
class CanvasPlan:
    found: bool
    brushes: list[Brush] = field(default_factory=list)
    cumulative: list[np.ndarray] = field(default_factory=list)
    best_gap: int = -1
    seconds: float = 0.0
    note: str = ""

    def text(self) -> str:
        head = (f"抽象画布解出 {len(self.brushes)} 笔" if self.found
                else f"抽象画布未解出(最好差 {self.best_gap} 格)")
        return f"[canvas] {head}, {self.seconds:.2f}s {self.note}"


def plan_canvas(start: np.ndarray, target: np.ndarray, brushes: list[Brush],
                width: int = 300, max_depth: int = 8) -> CanvasPlan:
    """在纯 numpy 画布上做图层分解。不碰游戏引擎, 快三个数量级。

    ⚠️起点用**答案区当前的真实内容**, 不是空白画布 —— 关卡开局答案区未必是空的,
    从空白起会算出一套根本执行不了的方案。
    """
    t0 = time.time()
    tgt = np.asarray(target)
    lanes = [(int((start != tgt).sum()), start, [])]
    best = int((start != tgt).sum())
    for _ in range(max_depth):
        nxt: dict[bytes, tuple[int, int, np.ndarray, list[int]]] = {}
        for _h, canvas, path in lanes:
            for i, b in enumerate(brushes):
                nc = b.apply(canvas)
                key = nc.tobytes()
                gap = int((nc != tgt).sum())
                if key in nxt and nxt[key][0] <= len(path) + 1:
                    continue
                nxt[key] = (len(path) + 1, gap, nc, path + [i])
        if not nxt:
            return CanvasPlan(False, best_gap=best, seconds=time.time() - t0,
                              note="画笔库空或无法再改变画布")
        ranked = sorted(nxt.values(), key=lambda x: (x[1], x[0]))
        best = min(best, ranked[0][1])
        if ranked[0][1] == 0:
            picked = [brushes[i] for i in ranked[0][3]]
            cum, acc = start.copy(), []
            for b in picked:
                cum = b.apply(cum)
                acc.append(cum.copy())
            return CanvasPlan(True, picked, acc, 0, time.time() - t0)
        lanes = [(c[1], c[2], c[3]) for c in ranked[:width]]
    return CanvasPlan(False, best_gap=best, seconds=time.time() - t0,
                      note=f"beam 宽 {width} 深 {max_depth} 未到 0")


def solve(game: Game, obs: Obs, st: CanvasSetup, target: np.ndarray,
          mask: np.ndarray | None = None, max_strokes: int = 10,
          max_seconds: float = 900.0, acts_fn=None, max_configs: int = 400,
          collect_seconds: float = 180.0) -> tuple[list[Action], Obs, str]:
    """闭环求解: **每落一笔就用实测画布重新规划**, 不开环执行整套方案。

    为什么不能开环: 实测在 cd82 L3 上, 抽象层 0.07 秒解出 4 笔, 真机把前
    3 笔都落对了, **第 4 笔找不到摆法** —— 抽象层与真机在最后一笔上脱节。
    开环拿到的信号是"找不到", 没有方向; 闭环每一笔之后都拿真实画布重来,
    脱节自动被吸收: 落完第 3 笔后重采的画笔库是**在新构型下**采的, 和
    在起始构型下采的不是一套。

    这也是"动作候选必须随状态动态重算"在画笔层面的同一句话。

    整个过程跑在克隆体上, 真机只走返回的这条序列。
    """
    t0 = time.time()
    box = st.answer_box
    node = game.fork()
    cur = obs
    full: list[Action] = []
    log: list[str] = []

    for k in range(max_strokes):
        if time.time() - t0 > max_seconds:
            return full, cur, f"超时, 落了 {k} 笔; " + "; ".join(log)
        canvas = _region(np.array(cur.grid), box)
        gap = int((canvas != target).sum())
        if gap == 0:
            return full, cur, f"画布已等于题面但未过关(判定不止看这块); " + "; ".join(log)

        # 🚨**每一笔都要重新做动作二分。** 第一版只在关卡开局 classify 一次,
        # 整个闭环都在用那份名单 —— 构型变了之后新出现的点击目标(正是那支
        # 能补上缺口的笔)名单里根本没有。cd82 L3 实测: 前三笔落对, 第四笔
        # 卡在**差 12 格**, 而"差 12"正是当年漏掉第二个面板时的同一个数字。
        #
        # 这是"动作候选必须随状态动态重算"的**第四次**出现(前三次: 搜索的
        # 候选表、槽搜索的固定动作表、可变格只在开局采)。
        # **凡是"这个游戏里有哪些动作可用/X 能不能被改变"的问题, 答案都是
        # 状态的函数, 不是关卡的常数。**
        #
        # ⚠️而且分类必须**跨状态取并集**: 判据"这次按下去改没改答案区"在
        # 第一笔涂完之后会失效 —— 再涂一次同样的内容就是无变化, 提交动作
        # 于是被重新归类成调整动作, 第二笔直接"采不到画笔"。
        # **当过提交动作就一直是提交动作。**
        if acts_fn is not None:
            fresh = classify(node, cur, acts_fn(cur), box)
            subs = {repr(a): a for a in st.submitters}
            subs.update({repr(a): a for a in fresh.submitters})
            adjs = {repr(a): a for a in st.adjusters}
            adjs.update({repr(a): a for a in fresh.adjusters})
            for k2 in subs:
                adjs.pop(k2, None)
            st = CanvasSetup(answer_box=box, submitters=list(subs.values()),
                             adjusters=list(adjs.values()))

        brushes, complete, judged, total, ncfg = collect_brushes(
            node, cur, st, mask, max_configs=max_configs,
            max_seconds=collect_seconds)
        if not brushes:
            return full, cur, f"第 {k+1} 笔: 采不到画笔; " + "; ".join(log)

        plan = plan_canvas(canvas, target, brushes)
        if plan.found:
            pick = plan.brushes[0]
            log.append(f"第{k+1}笔: 构型{ncfg}{'' if complete else '(截断)'} 笔{len(brushes)} 判{judged}/{total} -> 抽象层 {len(plan.brushes)} 笔到底, 先落第一笔")
        else:
            # 抽象层到不了底就走贪心: 挑这一笔之后差异最小的。
            # ⚠️贪心可能被"必须先变差"卡住, 但至少每一笔都有实测反馈,
            # 不会像开环那样走到最后才发现无解。
            best = min(brushes, key=lambda b: int((b.apply(canvas) != target).sum()))
            newgap = int((best.apply(canvas) != target).sum())
            if newgap >= gap:
                return full, cur, (f"第 {k+1} 笔: 没有任何一笔能缩小差异"
                                   f"(当前差 {gap}, 构型 {ncfg} 个{'' if complete else '**截断**'}, "
                                   f"笔 {len(brushes)} 支, 采集能判 {judged}/{total} 格); " + "; ".join(log))
            pick = best
            log.append(f"第{k+1}笔: 抽象层最好差 {plan.best_gap}, 贪心落一笔 {gap}->{newgap}")

        for a in pick.seq + [pick.submit]:
            cur = node.act(a)
            full.append(a)
            if cur.dead:
                return full, cur, f"第 {k+1} 笔后 GAME_OVER; " + "; ".join(log)
            if cur.level > obs.level:
                return full, cur, f"✅通关, {len(full)} 步; " + "; ".join(log)
    return full, cur, f"落满 {max_strokes} 笔仍未过关; " + "; ".join(log)


def execute(game: Game, obs: Obs, st: CanvasSetup, plan: CanvasPlan,
            mask: np.ndarray | None = None,
            per_stroke_seconds: float = 120.0) -> tuple[list[Action], Obs, str]:
    """把抽象计划翻译回真机: 逐笔搜"怎么调才能涂出累积画布该有的样子"。

    🚨判据必须用**累积画布**, 不能用单笔图案。踩过一次: 第 1 笔涂完后再提交
    得到的是两笔叠加, 跟抽象层记录的单笔图案永远对不上, 第 2 笔直接找不到。
    """
    box = st.answer_box
    full: list[Action] = []
    for k, want in enumerate(plan.cumulative):
        t0 = time.time()
        seen = {fingerprint(np.array(obs.grid), mask)}
        q: deque[tuple[list[Action], Game, Obs]] = deque([([], game.fork(), obs)])
        found = None
        while q and time.time() - t0 < per_stroke_seconds and found is None:
            seq, node, ob = q.popleft()
            for sub in st.submitters:
                got = _region(np.array(node.fork().act(sub).grid), box)
                if np.array_equal(got, want):
                    found = (seq, sub)
                    break
            if found:
                break
            for a in st.adjusters:
                child = node.fork()
                o = child.act(a)
                if o.dead:
                    continue
                fp = fingerprint(np.array(o.grid), mask)
                if fp in seen:
                    continue
                seen.add(fp)
                q.append((seq + [a], child, o))
        if found is None:
            return full, obs, f"第 {k+1} 笔在真机上找不到摆法(抽象层与真机脱节)"
        seq, sub = found
        for a in seq + [sub]:
            obs = game.act(a)
        full += seq + [sub]
        if obs.level > 0 and obs.dead:
            return full, obs, f"第 {k+1} 笔后 GAME_OVER"
    return full, obs, "全部笔已落"
