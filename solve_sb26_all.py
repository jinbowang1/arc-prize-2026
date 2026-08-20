"""sb26 通用解法: 把顶部那串目标**因式分解**到面板结构里, 逐关自动求解。

操作层(diag_sb26*.py / render_sb26.py 实测):
    点零件 = 排他性选中(20 格高亮); 选中后点某个空槽 = 放进去, 盘里那格清空
    A5 = 提交(摆对才过关);  A7 = 无作用
    第 53 行那条 64 格长条是动作预算, 每放一件消耗 1 格, 过关后回满

🚨语义层 —— 这才是这个游戏的题眼:
    顶部一排空心框 = **展开后**的目标序列
    带框面板       = 一段"程序"; 里面的 2 号小标记是空槽
    盘里的零件分两种(看是实心还是空心):
        实心块 = 内容, 放进槽里就往展开结果里贡献自己这一个颜色
        空心框 = **对子面板的一次调用**, 展开成那个同色子面板的全部内容
    所以同一个子面板可以被调用多次 —— L5 实测: 目标 9 项, 盘里只有 8 个零件,
    父面板 5 槽放 [6,@,@,11,15], @ 子面板 3 槽放 [14,8,8], 展开正好是
    6,14,8,8,14,8,8,11,15。两次 @ 共用同一份子面板内容。

求解 = 在"槽的容量固定 + 盘里零件恰好用完"的约束下, 找一组填法使展开等于目标。
关卡小, 直接回溯搜。
"""
from __future__ import annotations

import json

import numpy as np

from harness.env import Action, Game

MIN_PANEL_W = 10        # 面板边框的横线格数下限, 用来把面板和零件/目标框区分开
                        # (L3 起子面板并排, 边框只有 16 格宽; 零件一行才 4 格)


def components(mask: np.ndarray) -> list[np.ndarray]:
    """4-邻接连通块, 返回每块的坐标数组。"""
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    h, w = mask.shape
    for y0 in range(h):
        for x0 in range(w):
            if not mask[y0, x0] or seen[y0, x0]:
                continue
            stack, cells = [(y0, x0)], []
            seen[y0, x0] = True
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            out.append(np.array(cells))
    return out


def is_hollow(cells: np.ndarray) -> bool:
    """外接矩形比实际格数大 = 中间有洞 = 空心框(容器)。实心内容块两者相等。"""
    h = cells[:, 0].max() - cells[:, 0].min() + 1
    w = cells[:, 1].max() - cells[:, 1].min() + 1
    return int(h) * int(w) > len(cells)


def longest_run(row: np.ndarray) -> int:
    best = cur = 0
    for v in row:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


class Scene:
    """一帧里的全部结构。"""

    def __init__(self, grid):
        g = self.g = np.array(grid)
        h, w = g.shape
        self.bg = int(np.bincount(g.ravel()).argmax())

        # 预算条: 被某个非背景色占满的那一行
        # 取**最下面**那条: L5 起目标带的底色也铺满整行(第 0 行和第 7 行),
        # 从上往下找会把目标带当成进度条, 整个场景就全读歪了
        self.bar_row, self.bar_color = None, None
        for r in range(h):
            cnt = np.bincount(g[r], minlength=16)
            cnt[self.bg] = 0
            if cnt.max() >= w - 2:
                self.bar_row, self.bar_color = r, int(cnt.argmax())

        # 面板: 同一个色出现两条以上长横线, 上下两条就是它的边框
        self.panels = {}
        for c in np.unique(g):
            c = int(c)
            if c in (self.bg, self.bar_color):
                continue
            # 用整行格数而不是最长连续段: 子面板的管子会从父面板下边框穿过去,
            # 把那条横线打断成 16+10, 按最长段判会漏掉父面板(L2 实测)
            rows = [r for r in range(h) if int((g[r] == c).sum()) >= MIN_PANEL_W]
            # 🚨必须**成对**判, 不能拿 min/max: L5 的目标带里也有四个同色空心框,
            # 它们的两条边也够长, 直接取 min/max 会把面板上边界拉到第 1 行
            box = self._match_box(g, c, rows)
            if box:
                self.panels[c] = box

        # 盘: bar 行以下的非背景块
        self.tray = []
        if self.bar_row is not None:
            band = np.zeros((h, w), dtype=bool)
            band[self.bar_row + 1:] = True
            for c in np.unique(g[band]):
                if int(c) == self.bg:
                    continue
                for cc in components((g == c) & band):
                    self.tray.append({"color": int(c), "hollow": is_hollow(cc),
                                      "x": int(cc[:, 1].mean()),
                                      "y": int(cc[:, 0].mean())})
            self.tray.sort(key=lambda t: t["x"])

        # 目标序列: 最上面那条带里的空心框, 排除该带的底色(分隔色)
        top = min((p[0] for p in self.panels.values()), default=h)
        band = np.zeros((h, w), dtype=bool)
        band[:max(0, top - 2)] = True
        cnt = np.bincount(g[band], minlength=16)
        cnt[self.bg] = 0
        sep = int(cnt.argmax()) if cnt.max() else -1
        frames = []
        for c in np.unique(g[band]):
            if int(c) in (self.bg, sep):
                continue
            for cc in components((g == c) & band):
                if len(cc) >= 4:
                    frames.append((int(cc[:, 0].min()), int(cc[:, 1].mean()), int(c)))
        # 行优先: L8 的目标框排成两行, 只按 x 排会读成 [8,8,11,11,...]
        frames.sort()
        rows, cur = [], []
        for f in frames:
            if cur and f[0] - cur[0][0] > 3:
                rows.append(cur); cur = []
            cur.append(f)
        if cur:
            rows.append(cur)
        self.target = [c for row in rows for _, _, c in sorted(row, key=lambda f: f[1])]

    def _match_box(self, g, c, rows):
        """在候选横线里配出真正的面板边框: 上下两条列范围一致、左边框连得上、
        内部有空槽标记。目标带里的同色空心框满足前两条, 靠第三条排掉。"""
        pairs = sorted(((rb - ra, ra, rb) for i, ra in enumerate(rows)
                        for rb in rows[i + 1:] if rb - ra >= 3))
        for _, ra, rb in pairs:          # 先试最矮的框, 免得套住别的结构
            ca, cb = np.where(g[ra] == c)[0], np.where(g[rb] == c)[0]
            if ca.min() != cb.min() or ca.max() != cb.max():
                continue
            c0, c1 = int(ca.min()), int(ca.max())
            # 左边框要基本连满。松成 0.5 会出事: L5 里"目标带上边(第1行) +
            # 面板下边(第27行)"这一对列范围恰好相同, 左边框连通率 0.56 就蒙混过关
            if (g[ra + 1:rb, c0] == c).mean() < 0.9:
                continue
            if not (g[ra + 1:rb, c0 + 1:c1] == self.bar_color).any():
                continue
            return (ra, rb, c0, c1)
        return None

    def interior(self, color) -> np.ndarray:
        r0, r1, c0, c1 = self.panels[color]
        m = np.zeros_like(self.g, dtype=bool)
        m[r0 + 1:r1, c0 + 1:c1] = True
        return m

    def items(self, color) -> list[dict]:
        """面板里的格位: 空槽 + 已摆的零件, 按阅读顺序。

        🚨行序要按**上沿**算, 不能按平均 y: 引出子面板的零件拖着一条管子,
        平均 y 会被拽到同排空槽的后面, DFS 就退化成阅读序(L2 实测不过关)。
        """
        g, inner = self.g, self.interior(color)
        out = []
        for cc in components((g == self.bar_color) & inner):
            out.append({"kind": "slot", "y0": int(cc[:, 0].min()),
                        "x": int(cc[:, 1].mean()), "y": int(cc[:, 0].mean())})
        for c in np.unique(g[inner]):
            c = int(c)
            if c in (self.bg, self.bar_color):
                continue
            for cc in components((g == c) & inner):
                if len(cc) >= 9:
                    out.append({"kind": "piece", "color": c, "hollow": is_hollow(cc),
                                "y0": int(cc[:, 0].min()), "x": int(cc[:, 1].mean()),
                                "y": int(cc[:, 0].mean())})
        # 上沿相差 <=3 的算同一排, 排内按 x 走
        out.sort(key=lambda it: (it["y0"], it["x"]))
        rows, cur = [], []
        for it in out:
            if cur and it["y0"] - cur[0]["y0"] > 3:
                rows.append(cur); cur = []
            cur.append(it)
        if cur:
            rows.append(cur)
        return [it for row in rows for it in sorted(row, key=lambda i: i["x"])]

    def root(self):
        """根面板 = 没有被任何面板里的零件引出的那个。"""
        child = {it["color"] for c in self.panels for it in self.items(c)
                 if it["kind"] == "piece" and it["color"] in self.panels}
        cand = [c for c in self.panels if c not in child]
        return min(cand, key=lambda c: self.panels[c][0]) if cand else None

    def slots(self, panel) -> list[dict]:
        return [it for it in self.items(panel) if it["kind"] == "slot"]

    def emit(self, contents, limit) -> list[int]:
        """按"程序"跑出输出流, 最多 limit 项。

        🚨输出是可以**无限**的: L8 的正解是两个面板互相调用, 展开成周期串
        8,11,12,9,14,15,... 顶部那两行共 12 个框就是输出缓冲区, 填满即可。
        所以这里按 limit 截断, 不要求展开"跑完"。
        """
        out: list[int] = []
        panels = sorted(self.panels, key=lambda c: self.panels[c][0])
        cache = {c: self.items(c) for c in panels}

        def run(panel, depth):
            if depth > 64 or len(out) >= limit:
                return
            k = 0
            for item in cache[panel]:
                if len(out) >= limit:
                    return
                if item["kind"] == "piece":
                    key = (item["color"], item["hollow"])
                else:
                    key = contents[panel][k]
                    k += 1
                if key[1]:
                    if key[0] not in self.panels:
                        return
                    run(key[0], depth + 1)
                else:
                    out.append(key[0])

        run(panels[0], 0)
        return out

    def factor(self) -> dict[int, list[tuple[int, bool]]] | None:
        """枚举"每个槽放哪个零件", 跑一遍输出流跟目标对账。

        盘里的零件数恰好等于空槽数(八关皆然), 所以就是个全排列问题;
        重复颜色去重后规模很小(最大一关 9 个件、4 万种)。
        """
        panels = sorted(self.panels, key=lambda c: self.panels[c][0])
        if not panels or not self.target:
            return None
        slots = [(p, i) for p in panels for i in range(len(self.slots(p)))]
        if len(slots) != len(self.tray):
            return None
        inv: dict[tuple[int, bool], int] = {}
        for t in self.tray:
            inv[(t["color"], t["hollow"])] = inv.get((t["color"], t["hollow"]), 0) + 1
        contents = {p: [None] * len(self.slots(p)) for p in panels}

        def rec(i):
            if i == len(slots):
                return self.emit(contents, len(self.target)) == self.target
            p, k = slots[i]
            for key in list(inv):
                if inv[key] <= 0:
                    continue
                inv[key] -= 1
                contents[p][k] = key
                if rec(i + 1):
                    return True
                inv[key] += 1
            contents[p][k] = None
            return False

        return contents if rec(0) else None

def plan(sc: Scene) -> list[Action] | None:
    chosen = sc.factor()
    if not chosen:
        return None
    pool, seq = list(sc.tray), []
    for panel, keys in chosen.items():
        for slot, key in zip(sc.slots(panel), keys):
            cand = [t for t in pool if (t["color"], t["hollow"]) == key]
            if not cand:
                return None
            t = cand[0]
            pool.remove(t)
            seq += [Action.click(t["x"], t["y"], 6),
                    Action.click(slot["x"], slot["y"], 6)]
    if pool:                     # 盘里必须恰好用完
        return None
    seq.append(Action.key(5))
    return seq


def solve_level(game: Game, obs):
    sc = Scene(obs.grid)
    print(f"  面板={sorted(sc.panels)} 目标={sc.target} "
          f"盘={[t['color'] for t in sc.tray]}")
    seq = plan(sc)
    if seq is None:
        print("  ✗ 结构没读通")
        return None
    g = game.fork()                       # 先在克隆体上验, 真机不试错
    o = None
    for a in seq:
        o = g.act(a)
        if o.dead:
            print("  ✗ 克隆体 GAME_OVER")
            return None
    if o.level <= obs.level:
        print(f"  ✗ 克隆体走完没过关")
        return None
    print(f"  ✓ 克隆体验证通过: {len(seq)} 步")
    return seq


def main() -> None:
    game, obs = Game.make("sb26")
    baselines = [18, 28, 18, 19, 31, 23, 58, 18]
    all_seq, per_level = [], []
    while obs.level < obs.win_levels:
        lv = obs.level
        print(f"\n=== L{lv+1} (人类 {baselines[lv]}) ===")
        seq = solve_level(game, obs)
        if seq is None:
            break
        for a in seq:
            obs = game.act(a)
        all_seq += [repr(a) for a in seq]
        per_level.append(len(seq))
        print(f"  真机: L{lv+1} 过, 累计 {len(all_seq)} 步")

    done = obs.level
    print(f"\n=== 通关 {done}/{obs.win_levels} state={obs.state} | "
          f"AI {sum(per_level)} 步 vs 人类 {sum(baselines[:done])} ===")
    if done:
        json.dump({"game": "sb26", "seq": all_seq, "per_level_steps": per_level,
                   "baseline": baselines},
                  open("sb26_solutions.json", "w"), ensure_ascii=False, indent=1)
        print("已写 sb26_solutions.json")


if __name__ == "__main__":
    main()
