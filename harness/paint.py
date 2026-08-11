"""绘制类游戏求解器:枚举画笔 -> 抽象层图层分解 -> 翻译回真机。

适用形态: 有一个"提交"动作, 每次提交把某个形状涂进答案区, **后涂覆盖先涂**,
涂出题面即过关。cd82 就是这个形状; ft09 的填色、tr87 的答案区也沾边。

为什么必须分两层(cd82 L3 的教训):
  - 在真机上一步步搜: 逐步最佳优先 22439 节点 / 600 秒, h 卡在 22 不动;
    宏动作 beam 五条 lane 全卡 22。因为它要同时优化位置、倾斜、颜色和
    盖印时机, 是一次深指数。
  - 把画笔库抽出来后在**纯数组画布**上搜: 4 笔, **1 秒**解出。
  快了三个数量级, 而且不用 fork 游戏。

这正是 ls20 L6 那条定案的又一次应验: **真机负责"走得通不通", 离线模型
负责"该往哪走"**。

⚠️两个把我卡住过的坑, 都写在下面对应函数里: 画笔必须在**空画布**上采集;
翻译回真机时判据要用**累积画布**而不是单笔图案。
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .env import Action, Game, Obs
from .search import candidates, fingerprint


def _region(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, r1, c0, c1 = box
    return np.array(grid[r0:r1 + 1, c0:c1 + 1])


def extract_brushes(game: Game, base: Obs, keys: list[int], click_id: int | None,
                    submit: Action, answer_box: tuple[int, int, int, int],
                    mask: np.ndarray | None = None,
                    max_seconds: float = 420.0, blank: int = 0
                    ) -> tuple[dict[bytes, list[Action]], bool]:
    """枚举所有可达构型, 采集"按一次提交会涂出什么"。返回 (画笔, 是否采全)。

    🚨**这个函数采到的画笔是不完整的**, 只在答案区全空时才等于画笔本身。
    一笔可能把某些格子涂成"透明", 在空画布上看不出来, 但在非空画布上会
    **擦除**已有内容。cd82 上因此把一笔记成"只覆盖 12 格", 真身是覆盖
    50+ 格、其中 38 格涂透明 —— 抽象层据此排出的方案在真机上必然无解。
    要拿真实覆盖区, 用 `extract_brushes_two_tone`。

    ⚠️不要按"单色"过滤画笔。cd82 上这么滤过一次, 把 246 种里的 150 种多色
    画笔全丢了, 于是"没有任何一笔能落进单色区"看着像死局 —— 其实面板本身
    就可以是多色的。

    ⚠️第二个返回值是"队列有没有真的空"。**别把截断的采集当成完整的库**:
    库不全时抽象层会稳定收敛到一个非零差异, 看起来像"这关无解"。
    """
    t0 = time.time()
    adjust = [a for a in candidates(base, keys, click_id) if repr(a) != repr(submit)]
    seen = {fingerprint(np.array(base.grid), mask)}
    q: deque[tuple[list[Action], Game, Obs]] = deque([([], game.fork(), base)])
    out: dict[bytes, list[Action]] = {}
    while q and time.time() - t0 < max_seconds:
        seq, node, obs = q.popleft()
        painted = _region(np.array(node.fork().act(submit).grid), answer_box)
        if (painted != blank).any():
            key = painted.tobytes()
            if key not in out or len(seq) < len(out[key]):
                out[key] = seq
        for a in adjust:
            child = node.fork()
            o = child.act(a)
            if o.dead:
                continue
            fp = fingerprint(np.array(o.grid), mask)
            if fp in seen:
                continue
            seen.add(fp)
            q.append((seq + [a], child, o))
    return out, not q


def extract_brushes_two_tone(game: Game, floor_a: tuple[Game, Obs],
                             floor_b: tuple[Game, Obs],
                             keys: list[int], click_id: int | None, submit: Action,
                             answer_box: tuple[int, int, int, int],
                             mask: np.ndarray | None = None,
                             max_seconds: float = 420.0
                             ) -> dict[tuple[bytes, bytes], list[Action]]:
    """双底采集:拿到画笔的**真实覆盖区**, 含涂透明的部分。

    原理: 把答案区分别铺满两种不同底色, 从两个底走**同一条调整序列**(提交
    不改变构型, 所以同序列必到同构型), 各提交一次。
        两次结果相同的格子 = 被这一笔涂过(涂成什么与底无关);
        两次结果不同的格子 = 没被涂到(各自保留自己的底色)。
    这样"涂成透明"也能被看见, 覆盖区才是真的。

    返回 {(覆盖区掩码 bytes, 涂色 bytes): 到达序列}。
    """
    node_a, obs_a = floor_a
    node_b, obs_b = floor_b
    adjust = [a for a in candidates(obs_a, keys, click_id) if repr(a) != repr(submit)]

    t0 = time.time()
    seen = {fingerprint(np.array(obs_a.grid), mask)}
    q: deque[tuple[list[Action], Game, Obs]] = deque([([], node_a.fork(), obs_a)])
    seqs: list[list[Action]] = []
    while q and time.time() - t0 < max_seconds:
        seq, node, obs = q.popleft()
        seqs.append(seq)
        for a in adjust:
            child = node.fork()
            o = child.act(a)
            if o.dead:
                continue
            fp = fingerprint(np.array(o.grid), mask)
            if fp in seen:
                continue
            seen.add(fp)
            q.append((seq + [a], child, o))

    out: dict[tuple[bytes, bytes], list[Action]] = {}
    for seq in seqs:
        ca, cb = node_a.fork(), node_b.fork()
        for a in seq:
            ca.act(a)
            cb.act(a)
        ra = _region(np.array(ca.act(submit).grid), answer_box)
        rb = _region(np.array(cb.act(submit).grid), answer_box)
        covered = ra == rb
        if not covered.any():
            continue
        stroke = np.zeros_like(ra)
        stroke[covered] = ra[covered]
        key = (covered.tobytes(), stroke.tobytes())
        if key not in out or len(seq) < len(out[key]):
            out[key] = seq
    return out


@dataclass
class PaintPlan:
    strokes: list[np.ndarray] = field(default_factory=list)   # 每笔本身
    cumulative: list[np.ndarray] = field(default_factory=list)  # 每笔之后画布应有的样子
    seconds: float = 0.0

    def __len__(self) -> int:
        return len(self.strokes)


def plan_layers(brushes: dict[bytes, list[Action]], target: np.ndarray,
                shape: tuple[int, int], dtype, blank: int = 0,
                width: int = 400, max_depth: int = 8) -> PaintPlan | None:
    """在抽象画布上做图层分解。纯数组运算, 不碰游戏引擎, 快三个数量级。

    beam search: 状态 = 当前画布, 动作 = 一笔。目标 = 画布等于题面。
    """
    t0 = time.time()
    strokes = [np.frombuffer(k, dtype=dtype).reshape(shape) for k in brushes]
    tgt = np.asarray(target)

    def paint(canvas: np.ndarray, s: np.ndarray) -> np.ndarray:
        out = canvas.copy()
        m = s != blank
        out[m] = s[m]
        return out

    start = np.full(shape, blank, dtype=tgt.dtype)
    lanes = [(int((start != tgt).sum()), start, [])]
    for _ in range(max_depth):
        nxt: dict[bytes, tuple[int, int, np.ndarray, list[int]]] = {}
        for _h, canvas, path in lanes:
            for i, s in enumerate(strokes):
                nc = paint(canvas, s)
                key = nc.tobytes()
                if key in nxt and nxt[key][0] <= len(path) + 1:
                    continue
                nxt[key] = (len(path) + 1, int((nc != tgt).sum()), nc, path + [i])
        if not nxt:
            return None
        ranked = sorted(nxt.values(), key=lambda x: (x[1], x[0]))
        if ranked[0][1] == 0:
            picked = [strokes[i] for i in ranked[0][3]]
            cum, acc = np.full(shape, blank, dtype=tgt.dtype), []
            for s in picked:
                cum = paint(cum, s)
                acc.append(cum.copy())
            return PaintPlan(picked, acc, time.time() - t0)
        lanes = [(c[1], c[2], c[3]) for c in ranked[:width]]
    return None


def execute(game: Game, base: Obs, keys: list[int], click_id: int | None,
            submit: Action, answer_box: tuple[int, int, int, int],
            plan: PaintPlan, mask: np.ndarray | None = None,
            per_stroke_seconds: float = 150.0) -> tuple[list[Action], Obs]:
    """把抽象计划翻译成真机动作。逐笔在真机上搜"怎么摆才能涂出这一笔"。

    🚨判据必须用**累积画布**, 不能用单笔图案。踩过一次: 第 1 笔涂完后再提交
    得到的是两笔叠加, 跟抽象层记录的单笔图案永远对不上, 第 2 笔直接找不到。

    提交动作不改变画笔构型, 所以第 i+1 笔是从第 i 笔的构型接着调, 而不是
    从关卡起点重来 —— 这也是不能直接复用采集期那条到达序列的原因。
    """
    adjust = [a for a in candidates(base, keys, click_id) if repr(a) != repr(submit)]
    full: list[Action] = []
    obs = base

    for want in plan.cumulative:
        t0 = time.time()
        seen = {fingerprint(np.array(obs.grid), mask)}
        q: deque[tuple[list[Action], Game, Obs]] = deque([([], game.fork(), obs)])
        found = None
        while q and time.time() - t0 < per_stroke_seconds:
            seq, node, ob = q.popleft()
            got = _region(np.array(node.fork().act(submit).grid), answer_box)
            if np.array_equal(got, want):
                found = seq
                break
            for a in adjust:
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
            return full, obs
        for a in found:
            obs = game.act(a)
        obs = game.act(submit)
        full += found + [submit]
        if obs.level > base.level:
            break
    return full, obs
