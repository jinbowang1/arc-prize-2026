"""兜底穷举层:在克隆体上搜最短过关序列。

三条铁律固化在这里:

1. **状态指纹用语义态, 不用整帧像素**。整帧里有计数器/计时器/巡逻体, 同一
   位置在不同步数下永不相等, 去重失效, BFS 退化成穷举(ls20 实测内存吃到
   4GB)。这里的语义态 = 掩掉计数器格之后的帧。
2. **漏解可以, 错解不行**。语义去重可能合并掉本该区分的状态从而漏解; 但所有
   转移都在真模拟器上实走, 找到的路径按构造即已验证, 绝不会输出无效解。
3. **BFS 出最短路**。ARC 计分 RHAE=(人类步数/AI步数)^2, 步数就是分数。

动作候选每步重算: 幂等动作在状态变化后可能重新变得有效, 点击目标也随对象
移动而变。展开一个节点 = 对每个候选 peek 一次, 把无变化的剪掉。
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .env import Action, Game, Obs
from .percept import click_targets


def fingerprint(grid: np.ndarray, mask: np.ndarray | None,
                pending: int = 0) -> bytes:
    """状态指纹。

    🚨`pending` = 已注入但尚未结算的动作 id(见 `env.Game.act`)。**画面之外的
    隐藏状态必须进指纹。** sc25 上走 A3 和走 A1 之后画面完全相同(动作要下一次
    调用才结算), 只看画面就会把两者当成同一个状态全部去重, BFS 深度 0 就报
    "队列穷尽" —— 裸跑 1 秒结束。

    与"指纹不能用整帧像素(会混进计时器)"是同一条原则的两面: 指纹要正好等于
    **语义状态**, 多一点(计时器)会让去重失效, 少一点(输入缓冲)会让去重过头。
    """
    base = (grid * mask).tobytes() if mask is not None else grid.tobytes()
    return base if not pending else base + b"|p" + bytes([pending])


def candidates(obs: Obs, keys: list[int], click_id: int | None) -> list[Action]:
    """当前状态下的候选动作: 全部按键 + 塌缩后的点击目标。"""
    acts = [Action.key(i) for i in keys]
    if click_id is not None:
        acts += [Action.click(c, r, click_id) for (r, c) in click_targets(np.array(obs.grid))]
    return acts


@dataclass
class SearchResult:
    solved: bool
    seq: list[Action] = field(default_factory=list)
    nodes: int = 0
    expanded: int = 0
    seconds: float = 0.0
    reason: str = ""
    deepest: int = 0
    frontier: int = 0

    def text(self) -> str:
        head = f"解出, {len(self.seq)} 步" if self.solved else f"未解出({self.reason})"
        return (f"[search] {head} | 扩展 {self.expanded} 节点, 见过 {self.nodes} 状态, "
                f"最深 {self.deepest} 层, 队列剩 {self.frontier}, 用时 {self.seconds:.1f}s")


def bfs_level_up(game: Game, base: Obs, keys: list[int], click_id: int | None,
                 mask: np.ndarray | None = None,
                 max_depth: int = 100, max_nodes: int = 20000,
                 max_seconds: float = 120.0) -> SearchResult:
    """广度优先找"让 levels_completed 增加"的最短动作序列。

    预算上限来自 probe 探到的硬动作预算; 超时/超节点就交白卷往下走, 绝不
    把 ask_human 永远挡在后面。
    """
    t0 = time.time()
    seen = {fingerprint(np.array(base.grid), mask, base.pending)}
    # 队列元素: (动作序列, 到达该状态的 game 快照, 该状态的观测)
    q: deque[tuple[list[Action], Game, Obs]] = deque([([], game.fork(), base)])
    expanded = 0
    deepest = 0

    def done(reason: str) -> SearchResult:
        return SearchResult(False, nodes=len(seen), expanded=expanded, deepest=deepest,
                            frontier=len(q), seconds=time.time() - t0, reason=reason)

    while q:
        if time.time() - t0 > max_seconds:
            return done(f"超时 {max_seconds}s")
        if expanded >= max_nodes:
            return done(f"超节点上限 {max_nodes}")

        seq, node, obs = q.popleft()
        if len(seq) >= max_depth:
            continue
        expanded += 1
        deepest = max(deepest, len(seq))

        for a in candidates(obs, keys, click_id):
            child = node.fork()
            o = child.act(a)
            if o.dead:
                continue
            if o.level > base.level:
                return SearchResult(True, seq=seq + [a], nodes=len(seen), expanded=expanded,
                                    deepest=len(seq) + 1, frontier=len(q),
                                    seconds=time.time() - t0)
            fp = fingerprint(np.array(o.grid), mask, o.pending)
            if fp in seen:
                continue
            seen.add(fp)
            q.append((seq + [a], child, o))

    if deepest <= 1:
        return done("⚠️深度<=1 就队列穷尽 —— 几乎必然是状态指纹坏了"
                    "(掩码掩掉了游戏区域, 所有子状态指纹相同被全部去重), "
                    "不要当成'这关无解'")
    return done("队列穷尽(状态空间已封闭)")


def best_first(game: Game, base: Obs, keys: list[int], click_id: int | None,
               heuristic, mask: np.ndarray | None = None,
               max_depth: int = 100, max_nodes: int = 20000,
               max_seconds: float = 120.0,
               exclude: list[Action] | None = None) -> SearchResult:
    """用假设给出的距离做最佳优先搜索。

    `heuristic(node, obs) -> float` 必须**有梯度**。常数罚项会让这个函数
    退化成广度优先(ls20 L6 实测: h 在 11~24 之间横跳, 2 万次扩展未解;
    换成真实剩余步数后 6 秒解出)。传 node 而不只是网格, 是为了让启发式能
    隔着"提交动作"算距离(见 hypo.SubmitMatch)。

    `exclude` 把提交类动作移出分支。提交即判定的游戏里, 提交动作只该在
    每个节点试一次(启发式内部已经 peek 过), 放进分支只会让分支因子白涨。

    过关判定仍然只认引擎的 levels_completed —— 假设只用来决定搜索顺序,
    不用来判定是否通关。假设错了会慢, 但不会给出错解。
    """
    import heapq

    t0 = time.time()
    g0 = np.array(base.grid)
    seen = {fingerprint(g0, mask, base.pending)}
    counter = 0
    root = game.fork()
    skip = {repr(a) for a in (exclude or [])}
    heap: list[tuple[float, int, list[Action], Game, Obs]] = [
        (heuristic(root, base), 0, [], root, base)
    ]
    expanded = 0
    deepest = 0
    best_h = float("inf")

    def done(reason: str) -> SearchResult:
        return SearchResult(False, nodes=len(seen), expanded=expanded, deepest=deepest,
                            frontier=len(heap), seconds=time.time() - t0,
                            reason=f"{reason}, 最好的 h={best_h:.0f}")

    while heap:
        if time.time() - t0 > max_seconds:
            return done(f"超时 {max_seconds}s")
        if expanded >= max_nodes:
            return done(f"超节点上限 {max_nodes}")

        h, _, seq, node, obs = heapq.heappop(heap)
        best_h = min(best_h, h)
        if len(seq) >= max_depth:
            continue
        expanded += 1
        deepest = max(deepest, len(seq))

        # 提交动作单独试一次: 启发式已经 peek 过它, 这里只是把命中兑现成解
        for sub in (exclude or []):
            probe_child = node.fork()
            po = probe_child.act(sub)
            if po.level > base.level:
                return SearchResult(True, seq=seq + [sub], nodes=len(seen), expanded=expanded,
                                    deepest=len(seq) + 1, frontier=len(heap),
                                    seconds=time.time() - t0)

        for a in candidates(obs, keys, click_id):
            if repr(a) in skip:
                continue
            child = node.fork()
            o = child.act(a)
            if o.dead:
                continue
            if o.level > base.level:
                return SearchResult(True, seq=seq + [a], nodes=len(seen), expanded=expanded,
                                    deepest=len(seq) + 1, frontier=len(heap),
                                    seconds=time.time() - t0)
            cg = np.array(o.grid)
            fp = fingerprint(cg, mask, o.pending)
            if fp in seen:
                continue
            seen.add(fp)
            counter += 1
            # f = g + h, g 用步数 —— 保证解的步数不会太离谱(步数就是分数)
            heapq.heappush(heap, (len(seq) + 1 + heuristic(child, o), counter,
                                  seq + [a], child, o))

    return done("堆穷尽")
