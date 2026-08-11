"""交互探测层:开局用克隆体把"这个游戏怎么操作"问清楚,零真机成本。

探什么, 由三局的教训决定:

- **动作预算**(ft09 L6 第 128 击 GAME_OVER)。隐藏预算不探明, 后面所有实验
  结论整体作废。"速率好得不像话先怀疑游戏已死"。
- **无效动作**(ls20 被墙挡住: 仍扣资源但世界不推进)。noop 动作要从搜索
  分支里剔掉, 否则分支因子虚高。
- **状态环长**(tr87 七候选环)。可编辑对象常是个循环, 环长决定穷举底数。
- **互逆动作对**(tr87 ACTION1/2 精确互逆)。互逆意味着搜索里可以只走一个
  方向, 另一个方向用负步数表示。
- **决策帧 vs 动画帧**(通用)。通关动画/闪烁拒绝会污染 diff, 必须先分开。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .env import Action, Game, Obs


def grid(o: Obs) -> np.ndarray:
    return np.array(o.grid)


def probe_volatile(game: Game, base: Obs, keys: list[int],
                   clicks: list[Action] | None = None) -> np.ndarray:
    """找"每个动作都会改"的格子 —— HUD / 步数计数器 / 全局计时器。

    这些格子必须从 diff 与状态指纹里掩掉, 否则:
      - 任何两帧都不相等 -> 去重失效 -> BFS 退化成穷举(ls20 实测吃到 4GB)
      - "只动了计数器"的无效动作会被误判成有效动作

    判据: 一个格子是计数器, 当且仅当**它变成什么与你按了什么无关** ——
    每个动作后它都变, 且都变成同一个值(步数 +1 的显示)。携带动作信息的
    格子会因动作不同而变成不同的值, 那是游戏内容, 不能掩。

    两个更弱的判据都被实测推翻过, 别退回去:
      - 频率阈值: cd82 计数器只被点击触发, 按键不碰它, 混合路径的总体变化
        率够不到任何合理阈值。
      - 纯交集("每个动作都改的格子"): cd82 L2 所有动作恰好都改同一片游戏
        区域, 交集法把整片区域误判成计数器全掩掉, BFS 在深度 0 就报
        "状态空间已封闭"——搜索彻底瞎掉。

    ⚠️采样必须覆盖**全部动作类型**。只用按键采样会漏掉只被点击触发的计数器,
    于是 13 个实际无效的点击被误判成有效动作, 分支因子虚高 4 倍。

    返回布尔掩码, True = 该格可信(非易变)。
    """
    pool: list[Action] = [Action.key(i) for i in keys] + list(clicks or [])
    g0 = grid(base)
    changed_by_all: np.ndarray | None = None
    values: list[np.ndarray] = []
    for a in pool:
        o = game.peek(a)
        if o.dead:
            continue
        g1 = grid(o)
        d = g1 != g0
        if not d.any():
            continue
        changed_by_all = d if changed_by_all is None else (changed_by_all & d)
        values.append(g1)
    if changed_by_all is None or not changed_by_all.any() or len(values) < 2:
        return np.ones(g0.shape, dtype=bool)

    # 在"人人都改"的格子里, 只保留"改成的值处处相同"的那些 = 与动作无关
    stack = np.stack(values)
    action_independent = (stack == stack[0]).all(axis=0)
    counter = changed_by_all & action_independent
    if not counter.any():
        return np.ones(g0.shape, dtype=bool)
    return ~counter


def diff_cells(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> int:
    d = a != b
    if mask is not None:
        d = d & mask
    return int(d.sum())


def diff_bbox(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> tuple[int, int, int, int] | None:
    d = a != b
    if mask is not None:
        d = d & mask
    ch = np.argwhere(d)
    if len(ch) == 0:
        return None
    return (int(ch[:, 0].min()), int(ch[:, 0].max()),
            int(ch[:, 1].min()), int(ch[:, 1].max()))


@dataclass
class ActionProfile:
    """单个动作的效果签名。"""

    action: Action
    delta: int
    bbox: tuple[int, int, int, int] | None
    kills: bool = False          # 该动作直接导致 GAME_OVER
    advances_level: bool = False

    @property
    def is_noop(self) -> bool:
        return self.delta == 0 and not self.kills

    def line(self) -> str:
        if self.kills:
            return f"{self.action}: 致死"
        if self.is_noop:
            return f"{self.action}: 无效(帧无变化)"
        b = f" bbox={self.bbox}" if self.bbox else ""
        lv = " 过关!" if self.advances_level else ""
        return f"{self.action}: 变{self.delta}格{b}{lv}"


@dataclass
class ProbeReport:
    """探测层的全部产出。它同时是 ask_human() 报告的第一部分。"""

    game_id: str
    level: int
    kind: str
    profiles: list[ActionProfile] = field(default_factory=list)
    budget: int | None = None
    budget_note: str = ""
    cycles: dict[str, int] = field(default_factory=dict)
    inverse_pairs: list[tuple[str, str]] = field(default_factory=list)
    animation_frames: dict[str, int] = field(default_factory=dict)
    volatile_cells: list[tuple[int, int]] = field(default_factory=list)
    mask: np.ndarray | None = None   # True = 该格可信, 下游做指纹/diff 都要带上它

    @property
    def effective_keys(self) -> list[Action]:
        return [p.action for p in self.profiles if not p.is_noop and not p.kills]

    def text(self) -> str:
        out = [f"[probe] {self.game_id} L{self.level} 动作空间={self.kind}"]
        if self.volatile_cells:
            head = self.volatile_cells[:8]
            more = f" 等{len(self.volatile_cells)}格" if len(self.volatile_cells) > 8 else ""
            out.append(f"  易变格(已掩蔽, 疑似HUD/计时器): {head}{more}")
        for p in self.profiles:
            out.append("  " + p.line())
        if self.budget is not None:
            out.append(f"  动作预算: {self.budget} 步后 {self.budget_note}")
        else:
            out.append(f"  动作预算: 未探到上限 ({self.budget_note})")
        idem = [k for k, v in self.cycles.items() if v == 1]
        rings = {k: v for k, v in self.cycles.items() if v > 1}
        if idem:
            out.append("  幂等动作(按第二次无反应): " + ", ".join(idem))
        if rings:
            out.append("  状态环周期: " + ", ".join(f"{k}->{v}" for k, v in rings.items()))
        if self.inverse_pairs:
            out.append("  互逆动作对: " + ", ".join(f"{a}<->{b}" for a, b in self.inverse_pairs))
        if self.animation_frames:
            out.append("  含动画序列(引擎返回多层帧): " + str(self.animation_frames))
        return "\n".join(out)


def profile_actions(game: Game, base: Obs, keys: list[int],
                    mask: np.ndarray | None = None) -> list[ActionProfile]:
    """逐个键在克隆体上试一次, 记录效果签名(已掩掉易变格)。"""
    g0 = grid(base)
    out = []
    for i in keys:
        a = Action.key(i)
        o = game.peek(a)
        g1 = grid(o)
        out.append(ActionProfile(
            action=a,
            delta=diff_cells(g0, g1, mask),
            bbox=diff_bbox(g0, g1, mask),
            kills=o.dead,
            advances_level=o.level > base.level,
        ))
    return out


def probe_budget(game: Game, acts: list[Action], start_level: int = 0,
                 limit: int = 400) -> tuple[int | None, str]:
    """找隐藏的动作预算。ft09 L6 的第 128 击是 GAME_OVER。

    这类预算必须开局探明, 否则后面"扫全状态空间"的计划会在打一具尸体
    ("速率好得不像话先怀疑游戏已死")。

    用**两种**走法交叉验证, 因为单一动作撞死可能是那个动作的特定死法而
    非全局预算: (a) 连按第一个有效动作 (b) 轮流按所有有效动作。
    两者在同一步数上死 = 硬预算; 步数不同 = 与走法有关, 不是纯计步。

    ⚠️**过关不是预算耗尽**。预算的定义是"死亡前最多能走多少步"; 探测途中
    撞上过关只说明这条路走通了, 必须继续探。cd82 L2 上踩过: 探测第 1 步就
    过关 -> budget=1 -> BFS 的 max_depth 被设成 1 -> 只许搜一层 -> 报
    "状态空间已封闭", 看起来像无解, 其实是自己把手绑住了。
    """
    def run(seq_fn) -> tuple[int | None, str]:
        c = game.fork()
        crossed = 0
        for n in range(1, limit + 1):
            o = c.act(seq_fn(n))
            if o.dead:
                tail = f"(途中过关 {crossed} 次)" if crossed else ""
                return n, "GAME_OVER" + tail
            if o.level > start_level + crossed:
                crossed += 1          # 过关继续走, 不当预算上限
        tail = f"(途中过关 {crossed} 次)" if crossed else ""
        return None, f"{limit} 步内未死" + tail

    n1, why1 = run(lambda n: acts[0])
    n2, why2 = run(lambda n: acts[(n - 1) % len(acts)])
    if n1 == n2 and n1 is not None:
        return n1, f"{why1}(两种走法一致, 判定为硬动作预算)"
    if n1 is None and n2 is None:
        return None, f"两种走法各 {limit} 步均未触发"
    return (min(x for x in (n1, n2) if x is not None),
            f"单键={n1}({why1}) 轮换={n2}({why2}) — 步数不同, 与走法有关而非纯计步")


def _fp(g: np.ndarray, mask: np.ndarray | None) -> bytes:
    """状态指纹: 掩掉易变格后的帧。铁律——不能用整帧像素。"""
    return (g * mask).tobytes() if mask is not None else g.tobytes()


def probe_cycle(game: Game, act: Action, mask: np.ndarray | None = None,
                max_len: int = 32) -> int | None:
    """反复同一动作, 测状态多久回到原点。tr87 的七候选环就是这么测出来的。

    返回周期 n(>=2)。返回 1 表示**幂等**——按第二次世界没反应, 这跟"有个
    长度 1 的循环"是两回事, 报告里必须分开写, 否则会把"这键按了没用"
    读成"这键有个环"。返回 None = max_len 内没回到起点。
    """
    c = game.fork()
    start = _fp(grid(c.act(act)), mask)
    for n in range(2, max_len + 1):
        o = c.act(act)
        if o.dead:
            return None
        if _fp(grid(o), mask) == start:
            return n - 1
    return None


def probe_inverses(game: Game, base: Obs, keys: list[int],
                   mask: np.ndarray | None = None) -> list[tuple[str, str]]:
    """找互逆动作对: 走 a 再走 b 回到原状态。"""
    g0 = _fp(grid(base), mask)
    pairs = []
    for i in keys:
        for j in keys:
            if i == j:
                continue
            c = game.fork()
            c.act(Action.key(i))
            o = c.act(Action.key(j))
            if _fp(grid(o), mask) == g0:
                pair = tuple(sorted((f"A{i}", f"A{j}")))
                if pair not in pairs:
                    pairs.append(pair)
    return pairs


def probe_animation(game: Game, act: Action) -> int:
    """动画帧检测: 引擎这一步返回了几层帧。

    >1 表示这一步带动画序列, 中间层是动画帧。拿动画帧做 diff 会得出错误的
    动力学(Tycho 的决策帧/动画帧区分)。返回超出 1 的层数, 0 = 干净决策帧。
    """
    return max(0, game.peek(act).layers - 1)


def profile_clicks(game: Game, base: Obs, targets: list[tuple[int, int]],
                   click_id: int = 6, mask: np.ndarray | None = None) -> list[ActionProfile]:
    """逐个点击候选试一次。

    候选来自 percept 的连通块塌缩(4096 -> ~20), 这是 click 游戏的入场费。
    注意 percept 的坐标是 (row, col), 而引擎的 click data 是 (x, y) =
    (col, row) —— 这个顺序颠倒是 click 游戏最容易埋的一刀。
    """
    g0 = grid(base)
    out = []
    for (r, c) in targets:
        a = Action.click(c, r, click_id)
        o = game.peek(a)
        g1 = grid(o)
        out.append(ActionProfile(
            action=a,
            delta=diff_cells(g0, g1, mask),
            bbox=diff_bbox(g0, g1, mask),
            kills=o.dead,
            advances_level=o.level > base.level,
        ))
    return out


def run_probe(game: Game, base: Obs, kind: str, keys: list[int],
              clicks: list[Action] | None = None) -> ProbeReport:
    """完整探测流程。全程在克隆体上, 真机零消耗。

    `clicks` 由 percept.click_targets 塌缩而来。必须传进来 —— 计数器检测
    要覆盖全部动作类型, 漏掉点击就抓不到只被点击触发的计数器。
    """
    rep = ProbeReport(game_id=game.game_id, level=base.level, kind=kind)

    mask = probe_volatile(game, base, keys, clicks)
    rep.mask = mask
    rep.volatile_cells = [(int(r), int(c)) for r, c in np.argwhere(~mask)]

    rep.profiles = profile_actions(game, base, keys, mask)
    if clicks:
        rep.profiles += [p for p in profile_clicks(
            game, base, [(a.payload["y"], a.payload["x"]) for a in clicks], mask=mask)]

    live = [p.action for p in rep.profiles if not p.is_noop and not p.kills]
    rep.budget, rep.budget_note = probe_budget(
        game, live or [Action.key(keys[0])], start_level=base.level)

    for a in live:
        n = probe_cycle(game, a, mask)
        if n:
            rep.cycles[repr(a)] = n
    rep.inverse_pairs = probe_inverses(game, base, [a.id for a in live], mask)
    for a in live:
        m = probe_animation(game, a)
        if m:
            rep.animation_frames[repr(a)] = m
    return rep
