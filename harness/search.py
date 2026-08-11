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


def fingerprint(grid: np.ndarray, mask: np.ndarray | None) -> bytes:
    return (grid * mask).tobytes() if mask is not None else grid.tobytes()


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
    seen = {fingerprint(np.array(base.grid), mask)}
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
            fp = fingerprint(np.array(o.grid), mask)
            if fp in seen:
                continue
            seen.add(fp)
            q.append((seq + [a], child, o))

    if deepest <= 1:
        return done("⚠️深度<=1 就队列穷尽 —— 几乎必然是状态指纹坏了"
                    "(掩码掩掉了游戏区域, 所有子状态指纹相同被全部去重), "
                    "不要当成'这关无解'")
    return done("队列穷尽(状态空间已封闭)")
