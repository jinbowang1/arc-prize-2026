"""分解式模型:把动作分成"槽", 状态 = 每槽最后选了什么。

**为什么需要它(r11l L2 逼出来的):**

常量覆盖表那个偏置(见 model.py)在 r11l L1 成立、L2 全线失效 —— 45 个动作
全被判成"效果与状态有关"。但这不代表 L2 没结构, 只代表**结构不是"每个动作
盖一张固定的图"**。

结构诊断跑(diag_r11l_structure.py)在 L2 上量到:

    覆盖(a 之后按 b == 只按 b)  64%
    交换(ab == ba)                0%
    累积(两者都不是)             36%
    12×12 两步序列 -> 只有 31 个不同状态(全累积应是 144 个)

**64% 覆盖 + 状态数远小于序列数 = 槽结构。** 动作分成若干组, 同组内互相
覆盖(只有最后一次生效), 跨组则组合。这和 cd82 的"两个面板"是同一件事:
显示区由多个选择器共同决定, 单个动作只设定其中一个 —— 所以单看一个动作的
效果当然"与状态有关", 那是把组合当成了单体。

**0% 交换**说明还有绘制顺序: 后画的盖住先画的(cd82 "后涂覆盖先涂" 同款)。
所以槽之间不是完全独立, 顺序也是自由度, 搜索里要保留。

**这个表征的两个好处:**

1. 状态空间从 40^深度 塌到 ∏(槽大小)。
2. **每槽最多按一次** —— 同槽按两次是白按, 第一次会被第二次盖掉。于是解的
   长度上界 = 槽数, 而步数就是分数(RHAE 平方惩罚)。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .env import Action, Game, Obs


def _fp(o: Obs, mask: np.ndarray | None) -> bytes:
    g = np.array(o.grid)
    return (g * mask).tobytes() if mask is not None else g.tobytes()


@dataclass
class SlotModel:
    slots: list[list[Action]] = field(default_factory=list)
    loners: list[Action] = field(default_factory=list)   # 不与任何动作同槽
    dead: list[Action] = field(default_factory=list)
    noop: list[Action] = field(default_factory=list)
    override_rate: float = 0.0

    @property
    def combos(self) -> int:
        n = 1
        for s in self.slots:
            n *= len(s) + 1        # +1 = 这一槽不选
        return n * (len(self.loners) + 1)

    def text(self) -> str:
        sizes = [len(s) for s in self.slots]
        return (f"[factored] {len(self.slots)} 个槽 大小={sizes}, "
                f"独立动作 {len(self.loners)} 个, 无效 {len(self.noop)}, "
                f"致死 {len(self.dead)} | 覆盖率 {self.override_rate:.0%} | "
                f"组合数约 {self.combos}")


def learn_slots(game: Game, base: Obs, acts: list[Action],
                mask: np.ndarray | None = None,
                max_actions: int = 48) -> SlotModel:
    """按"互相覆盖"把动作分槽。

    判据: a 之后按 b, 结果等于只按 b —— 说明 b 把 a 的那部分状态盖掉了,
    两者写的是同一个槽。要求**双向**成立才归为一槽, 单向覆盖是别的关系
    (比如 b 的绘制面积恰好盖住 a), 不能当同槽。

    代价 = O(n²) 次 fork。n=40 时约 3200 次 ≈ 30 秒, 每关一次。换来的是
    搜索深度从"步数"降到"槽数"。
    """
    m = SlotModel()
    base_fp = _fp(base, mask)

    live: list[Action] = []
    single: dict[str, bytes] = {}
    for a in acts[:max_actions]:
        o = game.peek(a)
        if o.dead:
            m.dead.append(a)
            continue
        f = _fp(o, mask)
        if f == base_fp:
            m.noop.append(a)
            continue
        live.append(a)
        single[repr(a)] = f

    # override[i][j] = 走 i 再走 j 等于只走 j
    n = len(live)
    over = [[False] * n for _ in range(n)]
    pairs = 0
    hits = 0
    forks = [game.fork() for _ in live]
    for i, a in enumerate(live):
        ca = forks[i]
        ca.act(a)
        for j, b in enumerate(live):
            if i == j:
                continue
            pairs += 1
            if _fp(ca.fork().act(b), mask) == single[repr(b)]:
                over[i][j] = True
                hits += 1
    m.override_rate = hits / pairs if pairs else 0.0

    # 并查集: 双向覆盖才同槽
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if over[i][j] and over[j][i]:
                parent[find(i)] = find(j)

    groups: dict[int, list[Action]] = {}
    for i, a in enumerate(live):
        groups.setdefault(find(i), []).append(a)
    for g in groups.values():
        if len(g) > 1:
            m.slots.append(g)
        else:
            m.loners.extend(g)
    m.slots.sort(key=len, reverse=True)
    return m


@dataclass
class SlotResult:
    solved: bool
    seq: list[Action] = field(default_factory=list)
    states: int = 0
    forks: int = 0
    seconds: float = 0.0
    reason: str = ""
    best_h: float = float("inf")

    def text(self) -> str:
        head = (f"槽搜索解出 {len(self.seq)} 步" if self.solved
                else f"槽搜索未解出({self.reason})")
        return (f"[factored] {head} | 走过 {self.states} 个状态, {self.forks} 次 fork, "
                f"h 最好 {self.best_h:.0f}, {self.seconds:.1f}s")


def slot_search(game: Game, base: Obs, sm: SlotModel, distance=None,
                mask: np.ndarray | None = None, candidates=None,
                max_forks: int = 20000, max_seconds: float = 120.0) -> SlotResult:
    """在槽结构上搜索: **已知槽的动作每槽最多按一次**, 顺序保留(绘制有前后)。

    🚨`candidates(obs) -> list[Action]` **必须每个节点重算**。第一版拿开局
    算好的固定动作表跑, 在 r11l L2 上 0.7 秒"穷尽 261 种组合"报无解 ——
    而实测**走一步后新出现 86 个点击目标**(开局只有 70 个), 那张表里压根
    没有一多半的动作。这就是 cd82 L4 卡几小时的同一个错:
    **在起始态把动作候选算一次当全局事实, 等于假设只有一个东西会动。**
    结论来得又快又干脆的时候, 先怀疑表征。

    没被分过槽的新动作**不受"每槽一次"约束** —— 我们不知道它属于谁,
    宁可多搜也不要剪错。剪枝只作用在有证据的地方。

    完备性: 这个搜索完备当且仅当"同槽按两次没有意义"成立, 而那正是同槽的
    定义。若分槽分错会**漏解**。漏解可以错解不行 —— 过关判定只认引擎的
    levels_completed, 路径全在真模拟器上实走, 找到的解按构造即已验证。
    搜不到就交回上层, 别谎报无解。
    """
    t0 = time.time()
    slot_of: dict[str, int] = {}
    for i, s in enumerate(sm.slots):
        for a in s:
            slot_of[repr(a)] = i
    for k, a in enumerate(sm.loners):
        slot_of[repr(a)] = len(sm.slots) + k
    fixed = [a for s in sm.slots for a in s] + list(sm.loners)

    seen = {_fp(base, mask)}
    best = distance(np.array(base.grid)) if distance else float("inf")
    forks = 0
    unknown_seen = 0
    queue: list[tuple[list[Action], frozenset, Game, Obs]] = [([], frozenset(), game.fork(), base)]

    while queue:
        if time.time() - t0 > max_seconds:
            return SlotResult(False, states=len(seen), forks=forks,
                              seconds=time.time() - t0, best_h=best,
                              reason=f"超时 {max_seconds}s(其中 {unknown_seen} 个动作未分槽, 未剪枝)")
        if forks >= max_forks:
            return SlotResult(False, states=len(seen), forks=forks,
                              seconds=time.time() - t0, best_h=best,
                              reason=f"超 fork 上限 {max_forks}")
        seq, used, node, obs = queue.pop(0)
        acts = candidates(obs) if candidates else fixed
        for a in acts:
            si = slot_of.get(repr(a))
            if si is None:
                unknown_seen += 1     # 新出现的动作, 没有槽的证据, 不剪
            elif si in used:
                continue              # 同槽已按过, 再按会被盖掉, 白花一步
            child = node.fork()
            forks += 1
            o = child.act(a)
            if o.dead:
                continue
            if o.level > base.level:
                return SlotResult(True, seq=seq + [a], states=len(seen), forks=forks,
                                  seconds=time.time() - t0, best_h=0.0)
            f = _fp(o, mask)
            if f in seen:
                continue
            seen.add(f)
            if distance:
                best = min(best, distance(np.array(o.grid)))
            queue.append((seq + [a], used | ({si} if si is not None else set()),
                          child, o))

    return SlotResult(False, states=len(seen), forks=forks, seconds=time.time() - t0,
                      best_h=best,
                      reason=f"组合已穷尽(其中 {unknown_seen} 个动作未分槽, 未剪枝)")
