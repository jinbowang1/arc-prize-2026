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


UNKNOWN = -1     # 抽象画布上的"不知道"。真实颜色是 0..15, 所以它和任何目标都不等。


@dataclass
class Budget:
    """搜索预算: **主判据是确定性的(扩展节点数), 墙钟只当安全阀。**

    🚨**预算按秒给, A/B 对照就没有可比性** —— 同一份代码在机器忙的时候少搜
    一截, 走上完全不同的路。08-17 实测 (exp_canvas_openloop cd82 L3, 同一份
    基线代码跑两次):

        一次重采到构型 1107 -> 最终 "9 步, 差 22 格"
        一次被秒预算截到 879(截断) -> 最终 "13 步, 差 12 格"

    而仓库里这两个数字本来分别记在 fdf5626 和 f846d17 两次提交上, 当成了两次
    改动各自的效果 —— 至少有一部分是墙钟噪声。**判据挂在端到端结果上是对的,
    但端到端结果本身必须先可复现。**

    墙钟触发时 `wall_hit` 置位, 并且**必须一路报到日志**: 那一次的数字不可
    复现, 拿去跟别的跑做对照就是错的。安全阀只防跑飞, 不参与决定搜多深。
    """

    max_expansions: int = 4000
    wall_seconds: float = 1800.0
    expansions: int = 0
    t0: float = field(default_factory=time.time)
    wall_hit: bool = False

    def spend(self) -> bool:
        """花掉一次节点扩展。返回 False 表示预算到头, 该停了。"""
        if self.expansions >= self.max_expansions:
            return False
        if self.wall_expired():
            return False
        self.expansions += 1
        return True

    def wall_expired(self) -> bool:
        """只问安全阀。触发即记 wall_hit —— 这一位是"本次不可复现"的凭据。"""
        if time.time() - self.t0 > self.wall_seconds:
            self.wall_hit = True
            return True
        return False

    @property
    def note(self) -> str:
        return (" 🚨**墙钟安全阀触发, 本次结果不可复现, 不可用于对照**"
                if self.wall_hit else "")


@dataclass
class Brush:
    """一支笔: 走 seq 调整到某构型, 再按 submit, 会在答案区盖出 stroke。"""

    seq: list[Action]
    submit: Action
    covered: np.ndarray          # 布尔, 哪些格**有证据**被这一笔盖到
    stroke: np.ndarray           # 盖成什么色(只在 covered 上有意义)
    unknown: np.ndarray | None = None   # 这次采集**没有证据**的格
    cfg: bytes = b""             # 终点构型指纹 = 这支笔的**身份**(跨轮合并靠它)

    @property
    def size(self) -> int:
        return int(self.covered.sum())

    def apply(self, canvas: np.ndarray) -> np.ndarray:
        """涂一笔。**没有证据的格子涂成 UNKNOWN, 不是"保持原样"。**

        🚨这一条是 cd82 L3 卡在差 12 格的最后一环。此前 `apply` 把"采集判不了"
        默默当成了"这一笔不覆盖它" —— 于是抽象层能算出"第 4 笔涂完就全对",
        而真机上根本没有这支笔(execute 逐笔搜: 第 4 笔找不到摆法)。
        判不了的 50 格全都是被改变过的, 只是没有底能给出证据。

        标成 UNKNOWN 之后, 抽象层要么找到一条**处处有证据**的方案, 要么诚实地
        说未解出 —— 后者会把"该去补采集哪一块"直接指出来。
        **"漏解可以, 错解不行"** 在抽象层的同一句话。
        """
        out = canvas.copy()
        out[self.covered] = self.stroke[self.covered]
        if self.unknown is not None:
            out[self.unknown] = UNKNOWN
        return out


@dataclass
class CanvasSetup:
    answer_box: tuple[int, int, int, int]
    submitters: list[Action] = field(default_factory=list)
    adjusters: list[Action] = field(default_factory=list)
    # 只有显式 prune_noops(apply=True) 才会填这两个字段。默认不剔 —— 剔了
    # 可达构型会塌(1074 -> 56), 原委见 prune_noops 的实测表。
    noops: list[Action] = field(default_factory=list)
    pruned: bool = False

    def text(self) -> str:
        tail = f" | ⚠️已剔 {len(self.noops)} 条 noop 边" if self.pruned else ""
        return (f"[canvas] 答案区 {self.answer_box} | 提交动作 {len(self.submitters)} 个 "
                f"{[repr(a) for a in self.submitters[:4]]} | 调整动作 {len(self.adjusters)} 个"
                + tail)


def classify(game: Game, obs: Obs, acts: list[Action],
             answer_box: tuple[int, int, int, int]) -> CanvasSetup:
    """按"改不改得动答案区"把动作二分。

    ⚠️这里只有二分, 也只在**当前这一个状态**上分一次: 改得动答案区的是提交,
    **剩下一律算调整** —— 一个动作到底有没有用, 这个函数没问也答不出来。
    所以它的 adjusters 是"名义边表", 里头混着 noop(cd82 L3: 32 条里 15 条在
    开局附近什么都不改)。

    **但不要因此去剔它** —— 试过, 端到端更差, 实测记录见 `prune_noops`。
    BFS 自己会跳过 noop 边(构型指纹不变 -> `fp in seen`), 混进来只多花一次
    fork; 而剔错一条真边是漏解。想看水分有多大就用 `prune_noops` 的诊断模式。
    """
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


def _merge_brush(a: Brush, b: Brush) -> tuple[Brush, int]:
    """把同一支笔在两轮采集里的证据合起来。

    "同一支笔"的身份是 **(构型指纹, 提交动作)** —— 不是 (covered, stroke)。
    用后者当身份, 会把"A 轮判了上半区、B 轮判了下半区"的同一支笔拆成两支
    各带盲区的笔, 两轮挣来的证据永远合不到一起(实测: 底差异并集到 100/100,
    单笔盲区中位仍有 20 格, 抽象层照样解不出)。

    盲区取**交**(两轮都不知道才算不知道), 覆盖取**并**。
    返回冲突格数: 两轮都说盖到、却给出不同颜色的格子。**冲突不为 0 说明这支笔
    不是构型的函数**, 那是骨架("提交只按当前构型盖")出了问题, 必须报上去。
    """
    ua = a.unknown if a.unknown is not None else np.zeros_like(a.covered)
    ub = b.unknown if b.unknown is not None else np.zeros_like(b.covered)
    both = a.covered & b.covered
    conflict = int((a.stroke[both] != b.stroke[both]).sum())
    stroke = a.stroke.copy()
    stroke[b.covered] = b.stroke[b.covered]
    stroke[a.covered] = a.stroke[a.covered]      # 冲突时保留先采到的
    seq = a.seq if len(a.seq) <= len(b.seq) else b.seq
    return Brush(seq=list(seq), submit=a.submit, covered=a.covered | b.covered,
                 stroke=stroke, unknown=ua & ub, cfg=a.cfg), conflict


def _config_mask(game: Game, obs: Obs, st: CanvasSetup,
                 box: tuple[int, int, int, int], mask: np.ndarray | None,
                 spread: int = 6) -> np.ndarray:
    """构型掩码 = mask 再扣掉**提交动作会改动的、答案区之外的格子**。

    骨架那句话("提交只改画布, 调整只改构型")本身就给出了判据: 答案区之外
    任何被提交动作改动的格子, 按定义就不属于构型 —— 它是计数器/进度条。
    不用去认它是什么, 因果结构直接把它划出去。

    🚨这个洞的代价是完整的一条链: cd82 L3 上 probe 的跨步判据漏了 (63,55)
    这个落笔计数器(每涂一笔 4->5), 于是"提交改变了构型" -> 双底自证 10/10
    全被拒 -> 收不到底 -> 采集判 0/100 格 -> 画笔库瞎 -> 第 4 笔找不到摆法
    -> 卡在差 12 格。而它在**关卡开局测不出来**(开局提交, 答案区外 0 格变化),
    只有涂了几笔之后才现形。

    ⚠️必须在**多个构型**上问。只在当前构型问一次会漏掉"再提交一次才动"的格子 ——
    手工补掩 (63,55) 之后实测仍是 0/100, 就是因为污染格不止一个。
    "采样只在一个状态上做"在这个项目里已经栽到第六次了。
    """
    m = np.ones((64, 64), dtype=bool) if mask is None else mask.copy()
    r0, r1, c0, c1 = box
    nodes: list[tuple[Game, Obs]] = [(game.fork(), obs)]
    seen = {_config_fp(np.array(obs.grid), box, m)}
    # 摊开到若干个不同构型上, 每个都问一遍
    frontier = [(game.fork(), obs)]
    while frontier and len(nodes) < spread:
        nd, ob = frontier.pop(0)
        for a in st.adjusters:
            if len(nodes) >= spread:
                break
            ch = nd.fork()
            o = ch.act(a)
            if o.dead or o.level != obs.level:
                continue
            fp = _config_fp(np.array(o.grid), box, m)
            if fp in seen:
                continue
            seen.add(fp)
            nodes.append((ch, o))
            frontier.append((ch, o))

    touched = np.zeros((64, 64), dtype=bool)
    for nd, ob in nodes:
        g0 = np.array(ob.grid)
        for sub in st.submitters:
            o = nd.fork().act(sub)
            if o.dead or o.level != obs.level:
                continue
            touched |= (g0 != np.array(o.grid))
    touched[r0:r1 + 1, c0:c1 + 1] = False        # 答案区内被改是天经地义的
    return m & ~touched


def prune_noops(game: Game, obs: Obs, st: CanvasSetup,
                box: tuple[int, int, int, int], mask, spread: int = 8,
                apply: bool = False) -> tuple[int, int, int]:
    """统计边表里有多少条"在采样到的构型上什么都改不动"的边。

    返回 (疑似 noop 几条, 真边几条, 在几个构型上问过)。
    **默认只报告不动手。** `apply=True` 才真剔 —— 而下面的实测说明基本不该用。

    起因是个真问题: `classify` 只做二分, 剩下的一律进 adjusters。cd82 L3 上
    32 条名义边里 15 条在开局附近什么都不干(点在两个控件之间的缝上 ——
    `scene.targets` 把控件间隙也当成了可点目标), 任一构型上真实出度只有 14。

    🚨**但把它们剔掉是错的, 实测(08-17, exp_canvas_openloop cd82 L3):**

                    采集    画笔   可达构型   单笔盲区中位   抽象层
        不剔(基线)   143s   84 支   **1074**        0 格     解出 4 笔
        剔 15 条      10s   84 支   **56**         20 格   未解出(差 14)

    **可达构型塌掉 95%, 端到端从解出退成解不出。** 那 15 个动作在开局附近确实
    什么都不干, 走远了就不是了 —— 8 个采样构型听着"多", 占 1074 个只有 0.7%,
    而且是 BFS 最先探到的、彼此高度相似的一小撮。"采样只在一个状态上做"这次
    换的马甲是**"采样只在一小撮相邻状态上做"**: 样本量要相对**状态空间规模**
    衡量, 不是相对 1。

    🚨更根本的一笔账, 一开始就该算: **BFS 天然会跳过 noop 边** —— 走完构型
    指纹与父节点相同, `if fp in seen: continue` 当场跳掉。所以一条 noop 边的
    真实代价只是**一次 fork(线性常数)**, 而误剔一条真边的代价是**可达集塌缩**。
    风险与收益完全不对称, 拿指数去换常数, 方向本身就是反的。
    (分支因子"虚高 2.3 倍"这个数字没错, 错在它根本不是瓶颈。中间指标好看 ——
     采集快 14 倍、判 100/100 格 —— 端到端更差, 又一次。)

    所以这个函数的正当用途只剩**诊断**: 报出"名义边表里有多少水分", 给感知层
    (`scene.targets` 把控件间隙当目标)提改进线索, 而不是替搜索做剪枝决定。
    """
    if st.pruned:
        return 0, len(st.adjusters), 0

    # 采一批互不相同的构型当提问现场
    sites: list[tuple[Game, Obs]] = [(game.fork(), obs)]
    seen = {_config_fp(np.array(obs.grid), box, mask)}
    frontier = [(game.fork(), obs)]
    while frontier and len(sites) < spread:
        nd, ob = frontier.pop(0)
        for a in st.adjusters:
            if len(sites) >= spread:
                break
            ch = nd.fork()
            o2 = ch.act(a)
            if o2.dead or o2.level != ob.level:
                continue
            f = _config_fp(np.array(o2.grid), box, mask)
            if f in seen:
                continue
            seen.add(f)
            sites.append((ch, o2))
            frontier.append((ch.fork(), o2))

    live: set[str] = set()
    for nd, o in sites:
        base_cfg = _config_fp(np.array(o.grid), box, mask)
        base_canvas = _region(np.array(o.grid), box)
        for a in st.adjusters:
            r = repr(a)
            if r in live:
                continue                      # 已经证明有用了, 别再花 fork
            o2 = nd.peek(a)
            if o2.dead or o2.level != o.level:
                continue
            if _config_fp(np.array(o2.grid), box, mask) != base_cfg:
                live.add(r)
            elif not np.array_equal(_region(np.array(o2.grid), box), base_canvas):
                live.add(r)                   # 改画布不改构型 = 漏判的提交, 留着
    keep = [a for a in st.adjusters if repr(a) in live]
    drop = [a for a in st.adjusters if repr(a) not in live]
    if apply:
        # ⚠️只有在"采样构型数相对状态空间不算小"时才配这么做。上面的实测里
        # 8/1074 就把可达集砍掉了 95%。
        st.adjusters, st.noops, st.pruned = keep, drop, True
        print(f"[canvas] ⚠️边表剔 noop: {len(drop) + len(keep)} -> {len(keep)} 条 "
              f"(仅在 {len(sites)} 个构型上问过) {[repr(a) for a in drop[:6]]}", flush=True)
    elif drop:
        print(f"[canvas] 边表水分诊断: {len(drop)}/{len(drop) + len(keep)} 条边在这 "
              f"{len(sites)} 个构型上什么都不改(**没剔**) {[repr(a) for a in drop[:6]]}",
              flush=True)
    return len(drop), len(keep), len(sites)


def _trajectory_floors(game: Game, obs: Obs, st: CanvasSetup, anchor: list[Action],
                       box: tuple[int, int, int, int], mask, want: int = 6
                       ) -> tuple[Game, Obs, list[tuple[Game, np.ndarray]]] | tuple[None, None, list]:
    """第三种造底: 沿**同一条调整轨迹**走到同一个终点构型, 只在中途提交与否上不同。

    前两种造底在 cd82 L3 上都被实测堵死了:
      ① 在当前构型直接提交 —— 刚涂完的再涂一次是**幂等**的, 所有底一模一样
         (第 1 笔能判 65/100, 第 3 笔起判 **0/100**)
      ② 去别的构型涂一笔再走回来 —— 要求构型可逆, 而实测**构型图不强连通**:
         落 3 笔后 4 层 BFS 走不回开局构型(见过 88 个构型, 一个都不是)

    这条路两样都不要。依据只有一条已验事实: **提交不改构型**(cd82 实测 60/60)。
    于是两条轨迹只要调整序列逐步相同, 终点构型就必然相同, 而"中途涂了没涂"
    让画布不同 —— 正是双底判据要的那个前提。

    代价: 采集根从当前构型挪到了 anchor 终点, 所以每支笔的 seq 都要补上 anchor
    前缀, 真机多走 len(anchor) 步。

    ⚠️终点构型不靠假设, 逐个比对指纹自证; 对不上就不收这个底。
    """
    base = game.fork()
    cur = obs
    for a in anchor:
        cur = base.act(a)
        if cur.dead or cur.level != obs.level:
            return None, None, []
    anchor_cfg = _config_fp(np.array(cur.grid), box, mask)
    floors: list[tuple[Game, np.ndarray]] = [(base, _region(np.array(cur.grid), box))]

    # 在轨迹的哪些点位上提交 —— 枚举**点位组合**, 不是只插一笔。
    #
    # 🚨这一步就是把 cd82 当年手工的"答案区铺满两种底色"自动化。插一笔只能造出
    # 几十格的差异(实测开局判 65/100), 剩下的格子两底本来就一样, 采集给不出证据,
    # 画笔在那些格上的覆盖是猜的 —— 抽象层据此排出的第 4 笔真机上根本不存在。
    # 每个点位的构型不同 -> 涂的是不同的笔 -> 多涂几笔就能把答案区铺开。
    n = len(anchor)
    combos: list[tuple[int, ...]] = []
    for i in range(n + 1):
        combos.append((i,))                       # 单点
    if n >= 1:
        combos.append(tuple(range(n + 1)))        # 每个点位都涂: 铺得最满
        combos.append(tuple(range(0, n + 1, 2)))  # 隔点涂
        combos.append(tuple(range(1, n + 1, 2)))
    seen_floor = {_region(np.array(cur.grid), box).tobytes()}

    for combo in combos:
        for sub in st.submitters:
            if len(floors) >= want:
                return base, cur, floors
            f = game.fork()
            o = obs
            ok = True
            seq: list[Action] = []
            for i in range(n + 1):
                if i in combo:
                    seq.append(sub)
                if i < n:
                    seq.append(anchor[i])
            for a in seq:
                o = f.act(a)
                if o.dead or o.level != obs.level:
                    ok = False
                    break
            if not ok:
                continue
            if _config_fp(np.array(o.grid), box, mask) != anchor_cfg:
                continue          # 构型自证没过, 不收
            reg = _region(np.array(o.grid), box)
            if reg.tobytes() in seen_floor:
                continue          # 和已有的底一模一样, 收了也判不出新格子
            seen_floor.add(reg.tobytes())
            floors.append((f, reg))
    return base, cur, floors


class _ActionPool:
    """BFS 用的动作表 —— **随状态长, 不是关卡常数**。

    🚨坐标固定的点击在不同画面上点到的是不同东西, 所以"从这里能走到哪些构型"
    根本不是一张静态的图。cd82 L3 实测出过一个逻辑上不可能的结果:
    从当前构型 BFS 穷尽 56 个构型, 从 anchor 终点(就是从当前构型走过去的)
    穷尽 190 个 —— 静态图上可达集只会变小, 不可能涨。原因就是 BFS 全程在用
    关卡开局算出来的那份动作表。

    新冒出来的动作当场做二分, 判据和 `classify` 一样: 改得动答案区的是提交。
    ⚠️**当过提交就一直是提交**(跨状态取并集) —— 第一笔涂完再涂同样内容是
    无变化, 提交动作会被重新归类成调整, 下一笔直接采不到画笔。

    🚨**这里不做 noop 剔除**(试过在这里当场剔"改不动答案区也改不动构型"的新
    动作, 单独测下来对结果无影响, 而它是个单构型判断 —— 一个新目标在它刚出现
    的那个构型上没用, 完全可能在别处有用)。noop 边的代价只是一次 fork, BFS
    自己会跳过它; 剔错一条真边则是漏解。**多扩废节点只是慢, 掐窄可达集是漏解。**
    完整实测见 `prune_noops`。
    """

    def __init__(self, st: CanvasSetup, box: tuple[int, int, int, int], acts_fn):
        self.box = box
        self.acts_fn = acts_fn
        self.submit_keys = {repr(a) for a in st.submitters}
        self.adjusters: dict[str, Action] = {repr(a): a for a in st.adjusters}
        self.submitters: dict[str, Action] = {repr(a): a for a in st.submitters}
        # 多构型验过的 noop, 可以永久跳过, 连 peek 都省了
        self.noop_keys: set[str] = {repr(a) for a in st.noops}

    def at(self, node: Game, ob: Obs) -> list[Action]:
        if self.acts_fn is not None:
            before = _region(np.array(ob.grid), self.box)
            for a in self.acts_fn(ob):
                r = repr(a)
                # noop_keys 是 prune_noops 在多个构型上验过的, 才敢一直跳过
                if r in self.submit_keys or r in self.adjusters or r in self.noop_keys:
                    continue
                o2 = node.peek(a)
                if o2.dead or o2.level != ob.level:
                    continue
                if np.array_equal(_region(np.array(o2.grid), self.box), before):
                    self.adjusters[r] = a
                else:
                    self.submit_keys.add(r)
                    self.submitters[r] = a
        return list(self.adjusters.values())


def _sample(root: Game, robs: Obs, st: CanvasSetup, box: tuple[int, int, int, int],
            mask, floors: list[tuple[Game, np.ndarray]], prefix: list[Action],
            max_configs: int, budget: Budget, pool: "_ActionPool | None" = None,
            adj: dict[bytes, list[tuple[Action, bytes]]] | None = None
            ) -> tuple[list[Brush], bool, int, int, int]:
    """从 root 的构型 BFS 遍历构型 × 提交动作, 用 floors 双底判出每支笔的覆盖区。

    `adj` 非空时顺带把**构型图的边**记下来(构型 --动作--> 构型)。规划要用它:
    把"从上一笔的构型走不走得到这一笔"变成硬约束, 而不是挑完笔才发现走不过去。
    ⚠️指向"已经见过的构型"的边也要记 —— 那种边不产生新构型, 却正是回头路。
    """
    fp0 = _config_fp(np.array(robs.grid), box, mask)
    seen = {fp0}
    q: deque[tuple[list[Action], Game, Obs, bytes]] = deque([([], root.fork(), robs, fp0)])
    configs: list[tuple[list[Action], bytes]] = [([], fp0)]
    while q and len(configs) < max_configs and budget.spend():
        seq, node, ob, cur_fp = q.popleft()
        adjs = pool.at(node, ob) if pool is not None else st.adjusters
        for a in adjs:
            child = node.fork()
            o = child.act(a)
            if o.dead or o.level > robs.level:
                continue
            fp = _config_fp(np.array(o.grid), box, mask)
            if adj is not None:
                adj.setdefault(cur_fp, []).append((a, fp))
            if fp in seen:
                continue
            seen.add(fp)
            configs.append((seq + [a], fp))
            q.append((seq + [a], child, o, fp))
    complete = not q
    # 提交动作也可能是 BFS 途中才冒出来的, 用最新的全集去采笔
    submitters = list(pool.submitters.values()) if pool is not None else st.submitters

    # 哪些格子这次采集**有证据可判**: 底之间确实不同的格子。
    # 底都一样的格子, 这次采集给不出任何信息 —— 报出来的数字必须说清楚
    # 判了几格, 否则"覆盖 100 格"这种假消息会一路传到抽象层。
    bases = [reg for _, reg in floors]
    testable = np.zeros_like(bases[0], dtype=bool)
    for i in range(len(bases)):
        for j in range(i + 1, len(bases)):
            testable |= (bases[i] != bases[j])

    out: dict[tuple[bytes, bytes], Brush] = {}
    for seq, cfg in configs:
        # 采笔的规模已经被 max_configs 定死(确定性), 这里只留安全阀防跑飞
        if budget.wall_expired():
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
        for sub in submitters:
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
            # evidence = 这次采集对这一格**有话可说**的范围。
            # 范围之外既不能算覆盖, 也不能算"不覆盖" —— 它是未知, 必须往上报,
            # 否则抽象层会把"不知道"当成"不覆盖", 算出真机上不存在的方案。
            evidence = testable | changed
            covered = same & evidence
            if not covered.any():
                continue
            stroke = np.zeros_like(regs[0])
            stroke[covered] = regs[0][covered]
            unknown = ~evidence
            key = (covered.tobytes(), stroke.tobytes())
            cand = list(prefix) + list(seq)
            if key not in out or len(cand) < len(out[key].seq):
                out[key] = Brush(seq=cand, submit=sub, covered=covered,
                                 stroke=stroke, unknown=unknown, cfg=cfg)
    # 末位返回 testable 数组本身: 上层要拿它决定"下一轮该往哪造底"。
    # ⚠️只返回 sum 不够 —— 也不能用"跨笔的证据并集"代替它: 并集一开始就是满的,
    # 拿它当停止条件会让定向采集一轮都不跑(实测: 4 秒返回, 报 100/100, 而抽象层
    # 比认真采集过的那一版还差一倍)。**单笔盲区受 testable 约束, 并集不受。**
    return (list(out.values()), complete, int(testable.sum()),
            int(testable.size), len(configs), testable)


def collect_brushes(game: Game, obs: Obs, st: CanvasSetup,
                    mask: np.ndarray | None = None,
                    max_configs: int = 400, budget: Budget | None = None,
                    min_ratio: float = 0.9, acts_fn=None,
                    adj_out: dict[bytes, list[tuple[Action, bytes]]] | None = None
                    ) -> tuple[list[Brush], bool, int, int, int]:
    """枚举可达构型 × 提交动作, 双底采出每支笔的真实覆盖区。

    返回 (画笔库, 构型是否枚举完, 判了几格, 共几格, 构型数)。
    🚨**第二个返回值必须往上报。** 库不全时抽象层会稳定收敛到一个非零差异,
    看起来像"这关无解" —— 那是采集被截断, 不是游戏无解。

    预算走 `Budget`(确定性扩展数 + 墙钟安全阀), **不接受"多少秒"这种预算** ——
    原因见 `Budget` 的注释: 按秒给预算, 两次跑的结果没有可比性。

    造底分两档: 先用当前构型的底(便宜); **判得动的格子不到 min_ratio 就换
    轨迹底重采**(见 `_trajectory_floors`)。不无条件用轨迹底, 是因为它要真机
    多走 anchor 那几步 —— 判得动的时候没必要花这个钱。
    """
    budget = budget if budget is not None else Budget()
    box = st.answer_box

    # 🚨先把构型掩码算出来, 后面所有构型指纹都用它。probe 的掩码只保证掩掉了
    # 按步数走的计数器, 保证不了"每提交一次才动一格"的落笔计数器。
    mask = _config_mask(game, obs, st, box, mask)

    # 🚨这里**不要**调 prune_noops 去剔 noop 边。理由见那个函数的实测记录:
    # 剔完可达构型 1074 -> 56, 抽象层从"解出 4 笔"退成"解不出(差 14 格)"。

    # 多个底。构型都不变(提交不改构型), 只有画布内容不同。
    #
    # 🚨**判据的前提是"两底在这一格上本来就不同"。** 第一版只用两个底
    # (b = a 上按一次提交), 那一笔只改了几十格, 剩下的格子两底本来就一样,
    # 于是"结果也一样"被当成了"这一笔盖过" —— 报出**一支盖满整个 10×10
    # 答案区的笔**, 全是假的。当年手工铺两种纯色底是处处不同的, 自动版
    # 把这个前提丢了。
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

    # 动作池跨轮共用: BFS 途中新发现的点击目标, 后面几轮直接就能用上
    pool = _ActionPool(st, box, acts_fn)
    brushes, complete, judged, total, ncfg, tb = _sample(
        game, obs, st, box, mask, floors, [], max_configs, budget, pool, adj_out)
    if judged >= total * min_ratio:
        return brushes, complete, judged, total, ncfg

    # 判不动了 —— 换轨迹底, 而且**定向**挑 anchor。
    #
    # 🚨均摊式地随便挑几个调整动作当 anchor 是不行的。cd82 L3 开局实测, 可判性
    # 地图是一条完美的水平分界线: **上半 5 行全可判, 下半 5 行一格都判不了**,
    # 随便挑 anchor 只能把 50 抬到 65, 换三种造底方式数字纹丝不动。而那 50 格
    # 判不了的**全都被改变过**(够不着的 0 格) —— 缺的不是笔, 是"去涂那里的底"。
    # 后果: 第 4 笔那支笔在下半区的覆盖全是猜的, 抽象层据此算出"涂完就全对",
    # 真机上根本没有这支笔。
    #
    # 定向的做法: 拿第一档采到的库, 挑**覆盖到判不了区域**最多的笔, 用它的调整
    # 序列当 anchor —— 走到那个构型去涂一笔, 底就在判不了的地方有了差异。
    # (这些笔在判不了区的 covered 本身不可靠, 但拿来**排序**够用了 ——
    #  模型当排序器和当预测器是两条不同的及格线。)
    need = ~tb

    # 🚨各轮的笔要**合并成一个库**, 不能只留"最好的那一轮"。
    # 没有哪一支笔需要单独判满 —— 第一档的笔在上半区有证据, 定向轮的笔在下半区
    # 有证据, 抽象层完全可以用不同的笔去覆盖不同的区域。只留一轮等于把另一轮
    # 挣来的证据扔掉。
    scored = sorted(brushes, key=lambda b: -int((b.covered & need).sum()))
    cands: list[list[Action]] = [list(b.seq) for b in scored[:4]
                                 if (b.covered & need).any() and b.seq]
    cands += [[a] for a in st.adjusters[:2]]        # 兜底: 短序列

    # 身份 = (构型指纹, 提交动作)。同一支笔在不同轮采到的证据在这里合并。
    merged: dict[tuple[bytes, str], Brush] = {}
    for b in brushes:
        merged[(b.cfg, repr(b.submit))] = b
    tb_union = tb.copy()
    ncfg_total = ncfg
    conflicts = 0

    for anchor in cands:
        if budget.wall_expired():      # 安全阀; 轮数本身由 cands 定死
            break
        # 🚨停止条件用 **testable 的并集**, 不能用"跨笔的证据并集"。后者一开始
        # 就是满的(每格总有某支笔的 changed 说得清), 拿它当条件会让这个循环
        # 一轮都不跑 —— 实测 4 秒返回、报 100/100, 而抽象层比认真采集过的那版
        # 还差一倍(差 20 vs 差 10)。**好看的数字和有用的数字是两回事。**
        if int(tb_union.sum()) >= total * min_ratio:
            break
        root, robs, tfloors = _trajectory_floors(game, obs, st, anchor, box, mask)
        if root is None or len(tfloors) < 2:
            continue
        alt_b, alt_complete, _j, _t, alt_ncfg, alt_tb = _sample(
            root, robs, st, box, mask, tfloors, anchor, max_configs, budget, pool,
            adj_out)
        ncfg_total = max(ncfg_total, alt_ncfg)
        for b in alt_b:
            key = (b.cfg, repr(b.submit))
            if key in merged:
                merged[key], c = _merge_brush(merged[key], b)
                conflicts += c
            else:
                merged[key] = b
        complete = complete and alt_complete
        tb_union |= alt_tb
        need = ~tb_union          # 下一个 anchor 针对还没有底差异的地方

    if conflicts:
        # 不静默吞掉: 同一构型两轮给出不同颜色 = 笔不是构型的函数, 骨架有问题
        print(f"[canvas] ⚠️跨轮合并出现 {conflicts} 个冲突格 —— 笔可能不是构型的函数",
              flush=True)
    return list(merged.values()), complete, int(tb_union.sum()), total, ncfg_total


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


def _dists_from(cfg: bytes, adj: dict[bytes, list[tuple[Action, bytes]]],
                cache: dict[bytes, dict[bytes, int]]) -> dict[bytes, int]:
    """构型图上从 cfg 出发的单源最短距离(按调整动作步数)。结果缓存。"""
    if cfg in cache:
        return cache[cfg]
    dist = {cfg: 0}
    q: deque[bytes] = deque([cfg])
    while q:
        c = q.popleft()
        for _a, c2 in adj.get(c, ()):
            if c2 not in dist:
                dist[c2] = dist[c] + 1
                q.append(c2)
    cache[cfg] = dist
    return dist


def _path_on_graph(src: bytes, dst: bytes,
                   adj: dict[bytes, list[tuple[Action, bytes]]]) -> list[Action] | None:
    """在**采集时记下的构型图**上找路, 不在真机上重新 BFS。

    🚨"规划说可达、执行说走不到"的根源就在这里: 规划按 `adj` 算可达, 执行却用
    真机 BFS 重搜, 而两边的动作表是两个不同的 `_ActionPool` 实例 —— 采集时
    发现的动作, 执行时那个池子未必发现得了。同一张图问出两个答案。

    走图上的路是合法的, 因为骨架第二句实测成立(`diag_skeleton.py`: 同构型
    不同画布走同一条调整序列, 终点构型 **20/20 相同**) —— 调整动作的效果与
    画布无关, 所以采集时记下的边在执行时依然有效。
    """
    if src == dst:
        return []
    prev: dict[bytes, tuple[bytes, Action] | None] = {src: None}
    q: deque[bytes] = deque([src])
    while q:
        c = q.popleft()
        for a, c2 in adj.get(c, ()):
            if c2 in prev:
                continue
            prev[c2] = (c, a)
            if c2 == dst:
                path: list[Action] = []
                cur: bytes = dst
                while prev[cur] is not None:
                    p, act = prev[cur]          # type: ignore[misc]
                    path.append(act)
                    cur = p
                return list(reversed(path))
            q.append(c2)
    return None


def plan_canvas_graph(start_cfg: bytes, start: np.ndarray, target: np.ndarray,
                      brushes: list[Brush], adj: dict[bytes, list[tuple[Action, bytes]]],
                      width: int = 200, max_depth: int = 8
                      ) -> CanvasPlan:
    """在**构型图上**做图层分解 —— "走得过去"是硬约束, 不是事后才发现的问题。

    🚨`plan_canvas` 把画笔当成一个无序集合来挑, 挑完才在真机上发现走不过去
    (cd82 L3 实测: 抽象层解出 4 笔, 执行到第 2 笔报"构型图上走不到这支笔")。
    根子在于**根本没有一张静态的构型图**: 坐标固定的点击在不同画面上点到的是
    不同东西, 所以可达性依赖你站在哪。执行期补救只能救"走不到",
    救不了"顺序已经被前一笔破坏"。

    这里把可达性提进规划: 状态带上当前构型, 每一笔只能选**从当前构型走得到**的笔,
    代价 = 走过去的步数 + 提交那一步。于是排出来的方案在构型图上天然可执行。
    """
    t0 = time.time()
    tgt = np.asarray(target)
    cache: dict[bytes, dict[bytes, int]] = {}
    # (差异, 步数, 构型, 画布, 选中的笔)
    lanes = [(int((start != tgt).sum()), 0, start_cfg, start, [])]
    best = int((start != tgt).sum())
    for _ in range(max_depth):
        nxt: dict[tuple[bytes, bytes], tuple[int, int, bytes, np.ndarray, list[Brush]]] = {}
        for _gap, cost, cfg, canvas, path in lanes:
            dist = _dists_from(cfg, adj, cache)
            for b in brushes:
                d = dist.get(b.cfg)
                if d is None:          # 从这里走不到这支笔 —— 直接不是候选
                    continue
                nc = b.apply(canvas)
                key = (b.cfg, nc.tobytes())
                ncost = cost + d + 1
                if key in nxt and nxt[key][1] <= ncost:
                    continue
                nxt[key] = (int((nc != tgt).sum()), ncost, b.cfg, nc, path + [b])
        if not nxt:
            return CanvasPlan(False, best_gap=best, seconds=time.time() - t0,
                              note="构型图上没有可达的笔")
        ranked = sorted(nxt.values(), key=lambda x: (x[0], x[1]))
        best = min(best, ranked[0][0])
        if ranked[0][0] == 0:
            picked = ranked[0][4]
            cum, acc = start.copy(), []
            for b in picked:
                cum = b.apply(cum)
                acc.append(cum.copy())
            return CanvasPlan(True, picked, acc, 0, time.time() - t0,
                              note=f"(构型图上, 共 {ranked[0][1]} 步)")
        lanes = [(c[0], c[1], c[2], c[3], c[4]) for c in ranked[:width]]
    return CanvasPlan(False, best_gap=best, seconds=time.time() - t0,
                      note=f"构型图上 beam 宽 {width} 深 {max_depth} 未到 0")


def solve(game: Game, obs: Obs, st: CanvasSetup, target: np.ndarray,
          mask: np.ndarray | None = None, max_strokes: int = 10,
          wall_seconds: float = 900.0, acts_fn=None, max_configs: int = 400,
          collect_expansions: int = 4000) -> tuple[list[Action], Obs, str]:
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
        # 总控只是安全阀。真到了这里, 这次跑就不可复现了, 必须说出来。
        if time.time() - t0 > wall_seconds:
            return full, cur, ("🚨墙钟安全阀触发(本次结果不可复现, 不可用于对照), "
                               f"落了 {k} 笔; " + "; ".join(log))
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

        cb = Budget(max_expansions=collect_expansions, wall_seconds=wall_seconds)
        brushes, complete, judged, total, ncfg = collect_brushes(
            node, cur, st, mask, max_configs=max_configs, budget=cb)
        if not brushes:
            return full, cur, f"第 {k+1} 笔: 采不到画笔; " + "; ".join(log)

        plan = plan_canvas(canvas, target, brushes)
        if plan.found:
            pick = plan.brushes[0]
            log.append(f"第{k+1}笔: 构型{ncfg}{'' if complete else '(截断)'} 笔{len(brushes)} 判{judged}/{total} -> 抽象层 {len(plan.brushes)} 笔到底, 先落第一笔{cb.note}")
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


def _path_to_cfg(node: Game, obs: Obs, adjusters: list[Action], target_cfg: bytes,
                 box: tuple[int, int, int, int], mask, max_nodes: int = 6000,
                 pool: "_ActionPool | None" = None) -> list[Action] | None:
    """从当前构型 BFS 走到指定构型。搜的是构型图, 不是画布。

    ⚠️动作表同样要随状态重算 —— 否则会报"走不到", 而实际上是从开局那份名单里
    根本发不出通往那里的动作(cd82 L3 实测: 第 2 笔就报走不到)。
    """
    cur_fp = _config_fp(np.array(obs.grid), box, mask)
    if cur_fp == target_cfg:
        return []
    seen = {cur_fp}
    q: deque[tuple[list[Action], Game, Obs]] = deque([([], node.fork(), obs)])
    while q and len(seen) < max_nodes:
        seq, nd, ob = q.popleft()
        for a in (pool.at(nd, ob) if pool is not None else adjusters):
            ch = nd.fork()
            o = ch.act(a)
            if o.dead or o.level != obs.level:
                continue
            fp = _config_fp(np.array(o.grid), box, mask)
            if fp == target_cfg:
                return seq + [a]
            if fp in seen:
                continue
            seen.add(fp)
            q.append((seq + [a], ch, o))
    return None


def execute_cfg(game: Game, obs: Obs, st: CanvasSetup, plan: CanvasPlan,
                mask: np.ndarray | None = None, acts_fn=None
                ) -> tuple[list[Action], Obs, str]:
    """把抽象方案翻译回真机: **每一笔都从当前构型重新找路**。

    🚨这是 `execute` 的 bug 所在。画笔库里每支笔的 `seq` 都是从"采集时那个根
    构型"出发的; 连着用两支笔时, 第二支的 seq 从根走没有意义 —— 落完第一笔,
    构型已经停在第一笔那里了。`solve` 每落一笔整库重采所以撞不上; `execute`
    改用"在真机上搜一支能涂出目标画布的笔"来回避, 但那个 BFS 要从头搜到很深的
    构型, 180 秒里到不了, 报出来的是"第 4 笔找不到摆法" —— 看着像抽象层与真机
    脱节, 其实是路径找错了起点。

    有了 `Brush.cfg` 就不用绕: 直接在构型图上 BFS 到目标构型, 再按提交。
    提交不改构型, 所以落笔不会打乱这张图。

    ⚠️落完仍然逐笔核对实测画布 —— 对不上就停, 不硬走完。
    """
    box = st.answer_box
    node = game.fork()
    cur = obs
    full: list[Action] = []
    pool = _ActionPool(st, box, acts_fn)
    for k, b in enumerate(plan.brushes):
        path = _path_to_cfg(node, cur, st.adjusters, b.cfg, box, mask, pool=pool)
        if path is None:
            return full, cur, f"第 {k+1} 笔: 构型图上走不到这支笔的构型"
        for a in list(path) + [b.submit]:
            cur = node.act(a)
            full.append(a)
            if cur.dead:
                return full, cur, f"第 {k+1} 笔后 GAME_OVER"
            if cur.level > obs.level:
                return full, cur, f"✅通关, {len(full)} 步"
        got = _region(np.array(cur.grid), box)
        want = plan.cumulative[k]
        if not np.array_equal(got, want):
            return full, cur, (f"第 {k+1} 笔实测与预测差 {int((got != want).sum())} 格 "
                               f"(抽象层与真机在这一笔上脱节)")
    return full, cur, f"{len(plan.brushes)} 笔全部落对但未过关"


def solve_committed(game: Game, obs: Obs, st: CanvasSetup, target: np.ndarray,
                    mask: np.ndarray | None = None, max_strokes: int = 12,
                    wall_seconds: float = 900.0, acts_fn=None,
                    max_configs: int = 2000, collect_expansions: int = 6000
                    ) -> tuple[list[Action], Obs, str]:
    """**信任整套方案, 只在预测与实测分岔时才重规划。**

    对照 `solve`(每落一笔都重新采集+重新规划): 那一版会被贪心毁掉顺序。
    cd82 L3 实测 —— 卡住的 12 格是一个**必须先涂**的块, 库里有 5 支颜色全对的
    笔够得着, 但落下去会盖坏已涂好的部分(**后涂盖先涂**)。抽象层开局解出的
    4 笔方案本来含着正确顺序, 是逐笔重规划的贪心把顺序拆散的 ——
    贪心只看"这一笔之后差异最小", 而覆盖式涂色常常必须先让画面变差。

    当初改成逐笔重规划的理由是"开环第 4 笔找不到摆法", 而那次失败的真因是
    计数器 (63,55) 漏进构型指纹导致采集退化(见 `_config_mask`), 不是抽象层与
    真机脱节。前提修掉了, 结论也就不成立。

    分歧检测本身还是有用的: 它把"重规划"从每笔一次降到只在真出错时一次,
    并且**报出在第几笔脱节** —— 那是个诊断信号, 比默默重来值钱。

    ⚠️闭环省的是搜索预算, 不是真机步数。整个过程跑在克隆体上。
    """
    t0 = time.time()
    box = st.answer_box
    node = game.fork()
    cur = obs
    full: list[Action] = []
    log: list[str] = []
    plan: CanvasPlan | None = None
    idx = 0
    pool = _ActionPool(st, box, acts_fn)
    # 构型图跨轮累积 —— 重规划时上一轮探到的边依然有效(骨架②: 调整动作的
    # 效果与画布无关, diag_skeleton 实测 20/20)
    adj: dict[bytes, list[tuple[Action, bytes]]] = {}

    cb = Budget(max_expansions=collect_expansions, wall_seconds=wall_seconds)
    for k in range(max_strokes):
        # 总控只是安全阀。真到了这里, 这次跑就不可复现了, 必须说出来。
        if time.time() - t0 > wall_seconds:
            return full, cur, ("🚨墙钟安全阀触发(本次结果不可复现, 不可用于对照), "
                               f"落了 {k} 笔; " + "; ".join(log))
        canvas = _region(np.array(cur.grid), box)
        if int((canvas != target).sum()) == 0:
            return full, cur, "画布已等于题面但未过关(判定不止看这块); " + "; ".join(log)

        if plan is None or idx >= len(plan.brushes):
            # 只有这里会重新采集 —— 开局一次, 之后只在分歧后一次
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
            cb = Budget(max_expansions=collect_expansions, wall_seconds=wall_seconds)
            brushes, complete, judged, total, ncfg = collect_brushes(
                node, cur, st, mask, max_configs=max_configs,
                budget=cb, acts_fn=acts_fn, adj_out=adj)
            if not brushes:
                return full, cur, f"第 {k+1} 笔: 采不到画笔; " + "; ".join(log)
            # 在构型图上规划: "从上一笔走得到这一笔"是硬约束, 不是事后才发现的问题
            plan = plan_canvas_graph(_config_fp(np.array(cur.grid), box, mask),
                                     canvas, target, brushes, adj)
            idx = 0
            log.append(f"[规划] 构型{ncfg}{'' if complete else '(截断)'} 笔{len(brushes)} "
                       f"判{judged}/{total} -> "
                       f"{'解出 '+str(len(plan.brushes))+' 笔' if plan.found else '未解出(最好差 '+str(plan.best_gap)+')'}"
                       f"{cb.note}")
            if not plan.found:
                best = min(brushes, key=lambda b: int((b.apply(canvas) != target).sum()))
                if int((best.apply(canvas) != target).sum()) >= int((canvas != target).sum()):
                    return full, cur, (f"第 {k+1} 笔: 没有任何一笔能缩小差异"
                                       f"(当前差 {int((canvas != target).sum())}); " + "; ".join(log))
                plan = CanvasPlan(True, [best], [best.apply(canvas)], 0, 0.0)
                idx = 0

        pick = plan.brushes[idx]
        want = plan.cumulative[idx]
        # 🚨不能直接走 pick.seq —— 那条路是从**采集时那个根构型**出发的, 而现在
        # 站在上一笔停下的构型上。改成在构型图上从当前位置找路; 找不到就地重规划
        # (走不到本身就是一种分歧, 按闭环的规矩处理, 不当失败)。
        cur_cfg = _config_fp(np.array(cur.grid), box, mask)
        path = _path_on_graph(cur_cfg, pick.cfg, adj)
        if path is None:      # 图上没记到这条路, 才退回真机重搜
            path = _path_to_cfg(node, cur, st.adjusters, pick.cfg, box, mask, pool=pool)
        if path is None:
            log.append(f"[分歧] 第 {k+1} 笔: 从当前构型走不到这支笔 -> 重规划")
            plan = None
            continue
        for a in list(path) + [pick.submit]:
            cur = node.act(a)
            full.append(a)
            if cur.dead:
                return full, cur, f"第 {k+1} 笔后 GAME_OVER; " + "; ".join(log)
            if cur.level > obs.level:
                return full, cur, f"✅通关, {len(full)} 步; " + "; ".join(log)

        got = _region(np.array(cur.grid), box)
        if np.array_equal(got, want):
            idx += 1                      # 预测对上了, 接着走原方案, 不重采
        else:
            log.append(f"[分歧] 第 {k+1} 笔实测与预测差 {int((got != want).sum())} 格 -> 重规划")
            plan = None
    return full, cur, f"落满 {max_strokes} 笔仍未过关; " + "; ".join(log)


def execute(game: Game, obs: Obs, st: CanvasSetup, plan: CanvasPlan,
            mask: np.ndarray | None = None,
            per_stroke_expansions: int = 3000,
            wall_seconds: float = 600.0) -> tuple[list[Action], Obs, str]:
    """把抽象计划翻译回真机: 逐笔搜"怎么调才能涂出累积画布该有的样子"。

    🚨判据必须用**累积画布**, 不能用单笔图案。踩过一次: 第 1 笔涂完后再提交
    得到的是两笔叠加, 跟抽象层记录的单笔图案永远对不上, 第 2 笔直接找不到。
    """
    box = st.answer_box
    full: list[Action] = []
    for k, want in enumerate(plan.cumulative):
        # 每笔一份新预算, 确定性主判据 + 墙钟安全阀(见 Budget)
        bud = Budget(max_expansions=per_stroke_expansions, wall_seconds=wall_seconds)
        seen = {fingerprint(np.array(obs.grid), mask)}
        q: deque[tuple[list[Action], Game, Obs]] = deque([([], game.fork(), obs)])
        found = None
        while q and found is None and bud.spend():
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
