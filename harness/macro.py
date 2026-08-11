"""宏动作层:把"调整 + 提交"打包成一个宏, 用贪心逐块逼近目标。

为什么需要它: cd82 L3 用逐步最佳优先搜到 22439 节点、深度 43 层, h 卡在
28 不动 —— 因为一步一步搜要同时优化颜色、位置、旋转和提交时机, 这是一次
深指数。ls20 L7 的定案在这里同样成立: **把一次深指数拆成两次浅指数**。

做法:
  1. 在**不含提交动作**的空间里穷举所有可达调整状态(cd82 实测只有约 1300
     个, 一秒扫完)。这是第一次浅指数。
  2. 对每个状态 peek 一次提交动作, 看目标距离下降多少。挑性价比最高的执行。
     这是第二次浅指数。
  3. 重复, 直到通关或不再有改善。

⚠️挑选标准是**每步收益**(距离下降 / 宏长度)而不是纯距离下降 —— 步数就是
分数(RHAE 平方惩罚), 用 12 步换 5 步的收益是亏的。
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .env import Action, Game, Obs
from .search import candidates, fingerprint


@dataclass
class MacroResult:
    solved: bool
    seq: list[Action] = field(default_factory=list)
    rounds: int = 0
    seconds: float = 0.0
    trace: list[str] = field(default_factory=list)
    reason: str = ""

    def text(self) -> str:
        head = f"解出, {len(self.seq)} 步" if self.solved else f"未解出({self.reason})"
        return f"[macro] {head} | {self.rounds} 个宏, 用时 {self.seconds:.1f}s"


def reachable(game: Game, base: Obs, keys: list[int], click_id: int | None,
              submit: Action, mask: np.ndarray | None,
              max_states: int = 4000, max_seconds: float = 30.0,
              static_clicks: bool = True
              ) -> list[tuple[list[Action], Game, Obs]]:
    """穷举不含提交动作的可达状态。第一次浅指数。

    `static_clicks=True` 时点击候选只从起始帧算一次并复用。理由: 每个状态
    都重跑一遍连通块分析是这里的真瓶颈(cd82 实测 37 秒/轮几乎全花在这),
    而调色板这类点击目标在一关内位置固定。
    ⚠️代价是**可能漏解** —— 若某个点击目标是中途才出现的, 就抓不到。
    漏解可以错解不行: 所有转移仍在真模拟器上实走, 找到的路径按构造已验证。
    """
    t0 = time.time()
    seen = {fingerprint(np.array(base.grid), mask)}
    out: list[tuple[list[Action], Game, Obs]] = []
    q = deque([([], game.fork(), base)])
    fixed = candidates(base, keys, click_id) if static_clicks else None
    while q and len(seen) < max_states and time.time() - t0 < max_seconds:
        seq, node, obs = q.popleft()
        out.append((seq, node, obs))
        for a in (fixed if fixed is not None else candidates(obs, keys, click_id)):
            if repr(a) == repr(submit):
                continue
            child = node.fork()
            o = child.act(a)
            if o.dead:
                continue
            fp = fingerprint(np.array(o.grid), mask)
            if fp in seen:
                continue
            seen.add(fp)
            q.append((seq + [a], child, o))
    return out


def beam(game: Game, base: Obs, keys: list[int], click_id: int | None,
         submit: Action, distance, mask: np.ndarray | None = None,
         width: int = 4, max_rounds: int = 20, budget: int = 100,
         max_seconds: float = 300.0, verbose: bool = True) -> MacroResult:
    """宏动作 beam search。`distance(grid) -> float` 必须有梯度。

    为什么不是纯贪心: cd82 L3 上纯贪心把 h 从 100 推到 16 就卡死, 1201 个
    可达状态无一能改善 —— 最后那几格需要**先变坏再变好**(一次盖印会覆盖
    掉已经正确的部分)。beam 保留若干条暂时不占优的路线, 正是为这种情况。

    排序用 (距离, 步数): 距离优先, 同距离时步数少的优先 —— 步数就是分数。
    """
    t0 = time.time()
    # 每条候选: (距离, 已走序列, 节点, 观测)
    lanes = [(distance(np.array(base.grid)), [], game.fork(), base)]
    trace: list[str] = []
    best_seen = lanes[0][0]

    for rnd in range(max_rounds):
        if time.time() - t0 > max_seconds:
            return MacroResult(False, lanes[0][1], rnd, time.time() - t0, trace,
                               f"超时, 最好 h={best_seen:.0f}")

        nxt: list[tuple[float, list[Action], Game, Obs]] = []
        for h_now, seq, node, obs in lanes:
            for adj, child, o in reachable(node, obs, keys, click_id, submit, mask):
                if len(seq) + len(adj) + 1 > budget:
                    continue
                after = child.fork()
                po = after.act(submit)
                if po.dead:
                    continue
                if po.level > base.level:
                    return MacroResult(True, seq + adj + [submit], rnd + 1,
                                       time.time() - t0, trace + [f"R{rnd+1}: 通关"])
                nxt.append((distance(np.array(po.grid)), seq + adj + [submit], after, po))

        if not nxt:
            return MacroResult(False, lanes[0][1], rnd, time.time() - t0, trace,
                               f"无可扩展, 最好 h={best_seen:.0f}")

        # 去重必须用**状态指纹**。用 (距离, 步数) 当键踩过坑: beam 会被同一个
        # 状态的冗余走法占满, cd82 L3 上从第 3 轮起四条 lane 全是 h=22 只有
        # 步数在涨, 整个 beam 空转到超时。
        nxt.sort(key=lambda x: (x[0], len(x[1])))
        picked, sigs = [], set()
        for cand in nxt:
            sig = fingerprint(np.array(cand[3].grid), mask)
            if sig in sigs:
                continue
            sigs.add(sig)
            picked.append(cand)
            if len(picked) >= width:
                break
        lanes = picked
        best_seen = min(best_seen, lanes[0][0])
        line = f"R{rnd+1}: beam {[f'{c[0]:.0f}/{len(c[1])}步' for c in lanes]} ({time.time()-t0:.0f}s)"
        trace.append(line)
        if verbose:
            print("   " + line, flush=True)

    return MacroResult(False, lanes[0][1], max_rounds, time.time() - t0, trace,
                       f"达到最大轮数, 最好 h={best_seen:.0f}")
