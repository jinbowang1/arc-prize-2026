"""抽象模型层:把动作效果学成 numpy 里的一张表, 搜索快三个数量级。

**为什么必须有这一层(四局反复指向同一个缺口):**

- cd82 定案: 真机逐步搜 22439 节点 / 600 秒, h 卡在 22 不动; 把画笔抽出来
  在纯 numpy 画布上搜 —— **4 笔 1 秒解出**。
- ls20 L6 定案: "真机负责走得通不通, 离线模型负责该往哪走"。
- r11l 盲测再次撞上: 每扩展一个节点要 fork 38 个克隆体 × 10 毫秒 =
  **8 次扩展/秒**。BFS 90 秒只推到第 3 层。不是搜索不聪明, 是搜索太慢。

克隆体是正确性的底线, 但它太贵, 不能拿来当搜索的内循环。

**学什么: 最简偏置 —— 动作 a 的效果是一张固定的"格→色"覆盖表。**

这个假设在 cd82 的盖印、ft09 的填色上成立, 在 ls20 的移动、tr87 的环状态
推进上不成立。所以它**必须先验证再使用**: 同一个动作在多个不同状态下采样,
效果一致才登记, 不一致就明说"这个动作我建不了模", 交回真机。
🚨宁可说"建不了", 不能给一个会撒谎的模型 —— 开环规划吃的就是模型, 模型
错了整盘皆输, 而且失败信号出现得极晚(cd82 L4 走到第四笔才发现无解)。

**怎么采: 判据看"结果值"不看"变化量"。**

cd82 那条"画笔必须双底采集"的教训在这里被一般化了。当时在空画布上采笔,
"涂成透明"的格子看不见, 一笔被记成覆盖 12 格, 真身是覆盖 50+ 格擦掉 38 格,
抽象层据此排的方案真机必然无解。

根子在于**拿 diff 当效果**: 一个格子本来就是那个色, 盖上去 diff 为零, 于是
被漏掉。改成看结果值就自然消解了 —— 同一个动作在若干个不同状态下采样,
某个格子**每次采样后都变成同一个值**, 那这个值就是它盖出来的, 与之前是
什么色无关。不需要专门去铺两种底色, 只要采样的状态本身够不一样。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .env import Action, Game, Obs


@dataclass
class ActionModel:
    """一个动作的学到的效果。"""

    action: Action
    patch: dict[tuple[int, int], int] = field(default_factory=dict)
    support: int = 0            # 在几个不同状态下采过
    conflicts: int = 0          # 有几个格子在不同状态下结果不同
    kills: bool = False
    wins: bool = False          # 采样中出现过"这一步直接过关"

    @property
    def confident(self) -> bool:
        """够不够格进抽象搜索。

        conflicts>0 = 这个动作的效果与状态有关, 常量覆盖表是错的表征。
        support<2 = 只采过一个状态, 那不叫验证过, 只叫记录过一次。
        patch 为空 = 这个动作什么都不改, 它确实"建模成功"了, 但拿去搜索
        只会白占分支 —— 而且会让"可用动作 25 个"这种报告变成假消息:
        r11l L2 实测报出 25 个可用动作, 抽象层却在展开第 1 个节点后就堆
        穷尽, 因为那 25 个全是 noop。**空模型不算可用。**
        """
        return (self.support >= 2 and self.conflicts == 0
                and not self.kills and bool(self.patch))

    @property
    def noop(self) -> bool:
        return self.support >= 2 and self.conflicts == 0 and not self.patch

    def apply(self, grid: np.ndarray) -> np.ndarray:
        out = grid.copy()
        for (r, c), v in self.patch.items():
            out[r, c] = v
        return out

    def line(self) -> str:
        tag = "可用" if self.confident else ("致死" if self.kills
                                             else f"状态相关({self.conflicts}格冲突)"
                                             if self.conflicts else f"样本不足({self.support})")
        win = " 含过关" if self.wins else ""
        return f"{self.action}: 盖 {len(self.patch)} 格, 采样 {self.support} 次 [{tag}]{win}"


def distinct(states: list[tuple[Game, Obs]], mask: np.ndarray | None = None) -> int:
    """这批采样状态里有几个是真的互不相同。

    🚨"同一个动作在 4 个状态下效果一致"这句话, 只有当那 4 个状态真的不一样
    时才是验证; 如果游走走的全是 noop, 4 个状态其实是同一个, 那就是把一个
    样本数了四遍, 然后管它叫"验证通过"。**同源的两个结论互相印证等于没印证**
    的第三次现形(前两次: cd82 抽象层与真机 beam 都算出差 12 格)。
    """
    keys = {((np.array(o.grid) * mask) if mask is not None else np.array(o.grid)).tobytes()
            for _, o in states}
    return len(keys)


def collect_states(game: Game, base: Obs, actions: list[Action],
                   n: int = 4) -> list[tuple[Game, Obs]]:
    """取若干个**互不相同**的状态用于采样。

    状态越不一样, "结果值处处相同"这个判据就越强 —— 它替代了 cd82 那次
    手工铺两种底色的做法。这里用不同长度的确定性游走来制造差异(不用随机数,
    同一局重跑结果要能复现)。
    """
    out = [(game.fork(), base)]
    if not actions:
        return out
    for i in range(1, n):
        c = game.fork()
        o = base
        for k in range(i):
            a = actions[(i * 3 + k) % len(actions)]
            nxt = c.act(a)
            if nxt.dead or nxt.level > base.level:
                break        # 死了或过关了就别再往下走, 那不是同一关的状态
            o = nxt
        out.append((c, o))
    return out


def learn(game: Game, base: Obs, actions: list[Action], n_states: int = 4,
          mask: np.ndarray | None = None) -> dict[str, ActionModel]:
    """学每个动作的覆盖表。全程在克隆体上, 真机零消耗。

    代价 = len(actions) × n_states 次 fork。r11l 上 38 × 4 ≈ 150 次 ≈ 1.5 秒,
    换来的是之后每个节点微秒级 —— 这笔账在第一百个节点就回本了。
    """
    states = collect_states(game, base, actions, n_states)
    n_distinct = distinct(states, mask)
    if n_distinct < 2:
        # 采不出两个不同的状态就没法做"效果与状态无关"的验证。这时候宁可
        # 交白卷, 也不能报一堆"可用"。
        return {repr(a): ActionModel(action=a, support=n_distinct) for a in actions}
    models: dict[str, ActionModel] = {}

    for a in actions:
        m = ActionModel(action=a)
        afters: list[np.ndarray] = []
        befores: list[np.ndarray] = []
        for c, o in states:
            child = c.fork()
            r = child.act(a)
            # 🚨滞后局要走两次才看得到 a 的效果。这类游戏 perform_action 只把动作
            # 放进缓冲、下一次调用才结算(见 env.Game.act), 单步看到的是**上一个**
            # 动作的效果 —— sc25 上因此 27 个动作全报"改 0 格"。
            # 走两次得到的正是 a 的**单次**真效果(实测 viaact(3,3) 逐格等于
            # truth(3)), 与 Game.effect 同一套语义; 克隆体试探免费。
            # ls20/cd82 这类即时生效的局 lagged=False, 一次都不多花。
            if not r.dead and getattr(c, "lagged", False):
                r = child.act(a)
            if r.dead:
                m.kills = True
                break
            if r.level > o.level:
                m.wins = True
                continue      # 过关帧属于下一关, 不能拿来学本关的效果
            befores.append(np.array(o.grid))
            afters.append(np.array(r.grid))
        if m.kills or len(afters) < 1:
            models[repr(a)] = m
            continue

        # support 记的是**互不相同的**采样状态数, 不是采样次数
        m.support = min(len(afters), n_distinct)
        # 只看"至少在某次采样里变过"的格子。没变过的格子给不出信息 ——
        # 它可能不归这个动作管, 也可能归它管但恰好本来就是那个色。
        touched = np.zeros(afters[0].shape, dtype=bool)
        for b, aft in zip(befores, afters):
            touched |= (b != aft)
        if mask is not None:
            touched &= mask       # 计数器/HUD 不进模型

        stack = np.stack(afters)
        for r, c in np.argwhere(touched):
            vals = stack[:, r, c]
            if len(set(vals.tolist())) == 1:
                m.patch[(int(r), int(c))] = int(vals[0])
            else:
                # 同一个动作在不同状态下把这格涂成不同的色 = 效果与状态有关,
                # 常量覆盖表在这里就是假的。记下来, 别硬编一个值进去。
                m.conflicts += 1
        models[repr(a)] = m
    return models


def coverage(models: dict[str, ActionModel]) -> str:
    ok = [m for m in models.values() if m.confident]
    dep = [m for m in models.values() if m.conflicts]
    kil = [m for m in models.values() if m.kills]
    nop = [m for m in models.values() if m.noop]
    thin = [m for m in models.values() if not m.confident and not m.conflicts
            and not m.kills and not m.noop]
    cells = sorted((len(m.patch) for m in ok), reverse=True)[:3]
    return (f"[model] {len(models)} 个动作: 可用 {len(ok)}(最大覆盖 {cells}) / "
            f"状态相关 {len(dep)} / 无效果 {len(nop)} / 致死 {len(kil)} / "
            f"样本不足 {len(thin)}")


@dataclass
class Plan:
    found: bool
    seq: list[Action] = field(default_factory=list)
    nodes: int = 0
    seconds: float = 0.0
    reason: str = ""
    best_h: float = float("inf")

    def text(self) -> str:
        head = f"抽象层解出 {len(self.seq)} 步" if self.found else f"抽象层未解出({self.reason})"
        return f"[plan] {head} | 展开 {self.nodes} 节点, h 最好 {self.best_h:.0f}, {self.seconds:.1f}s"


def plan(grid0: np.ndarray, models: dict[str, ActionModel], distance,
         max_depth: int = 12, max_nodes: int = 200000,
         max_seconds: float = 30.0) -> Plan:
    """在 numpy 画布上做最佳优先搜索。只用 confident 的动作。

    ⚠️抽象层**不判过关** —— 过关只有引擎说了算。这里判的是"启发式距离到 0",
    那只是一个候选方案。它必须回真机走一遍才算数(见 closedloop)。
    ls20 L6 的边界重复陷阱就是"搜索内通关 ≠ 解可复现"的老账。
    """
    import heapq
    import time

    usable = [m for m in models.values() if m.confident]
    if not usable:
        return Plan(False, reason="没有任何动作建成了常量覆盖表 —— 这个游戏的"
                                  "效果与状态有关, 抽象层用不上, 该回真机搜")
    t0 = time.time()
    seen = {grid0.tobytes()}
    counter = 0
    h0 = distance(grid0)
    heap: list[tuple[float, int, list[Action], np.ndarray]] = [(h0, 0, [], grid0)]
    nodes = 0
    best = h0

    while heap:
        if time.time() - t0 > max_seconds:
            return Plan(False, nodes=nodes, seconds=time.time() - t0,
                        reason=f"超时 {max_seconds}s", best_h=best)
        if nodes >= max_nodes:
            return Plan(False, nodes=nodes, seconds=time.time() - t0,
                        reason=f"超节点 {max_nodes}", best_h=best)
        h, _, seq, g = heapq.heappop(heap)
        best = min(best, h - len(seq))
        nodes += 1
        if len(seq) >= max_depth:
            continue
        for m in usable:
            g2 = m.apply(g)
            key = g2.tobytes()
            if key in seen:
                continue
            seen.add(key)
            d = distance(g2)
            if d <= 0:
                return Plan(True, seq=seq + [m.action], nodes=nodes,
                            seconds=time.time() - t0, best_h=0.0)
            counter += 1
            heapq.heappush(heap, (len(seq) + 1 + d, counter, seq + [m.action], g2))

    return Plan(False, nodes=nodes, seconds=time.time() - t0,
                reason="堆穷尽(可用动作太少或效果不足以到达目标)", best_h=best)


def predictor(models: dict[str, ActionModel]):
    """给 closedloop.run 用的 predict(i, obs, a) -> 预测网格 | None。

    模型没把握的动作返回 None(跳过校验)而不是瞎猜一个 —— 校验层要能区分
    "我预测错了"和"我根本没预测"。
    """
    def predict(i: int, obs: Obs, a: Action):
        m = models.get(repr(a))
        if m is None or not m.confident:
            return None
        return m.apply(np.array(obs.grid))
    return predict
