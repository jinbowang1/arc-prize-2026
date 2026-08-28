"""只戴归因表这副墨镜打游戏 —— 模拟模型的处境。
能看: 归因表的列(帧数/变化格数/bbox/颜色迁移/点击处颜色) + 当前棋盘 ascii。
不能看: 源码、引擎内部状态、克隆体试错。用真实步数。
"""
import os, sys
os.environ.setdefault("OPERATION_MODE", "OFFLINE")
os.chdir(os.path.expanduser("~/Desktop/project/arc-agi-3"))
import numpy as np, arc_agi
from arcengine import GameAction

# 跟 solver.py 里归因表用的是同一套算法, 这里独立一份避免 import 整个 harness
ARC_COLOR_CHARS = "WwgGcBMPRbSYOrNp"

def _cell_color_char(grid, row, col):
    try: r, c = int(row), int(col)
    except (TypeError, ValueError): return "?"
    if r < 0 or r >= len(grid): return "?"
    line = grid[r]
    if c < 0 or c >= len(line): return "?"
    return ARC_COLOR_CHARS[max(0, min(15, int(line[c])))]

def _describe_change(before, after):
    cells = []
    for r in range(max(len(before), len(after))):
        rb = before[r] if r < len(before) else ()
        ra = after[r] if r < len(after) else ()
        for c in range(max(len(rb), len(ra))):
            vb = rb[c] if c < len(rb) else None
            va = ra[c] if c < len(ra) else None
            if vb != va: cells.append((r, c, vb if vb is not None else -1, va if va is not None else -1))
    if not cells: return {"changed": 0}
    rows = [x[0] for x in cells]; cols = [x[1] for x in cells]
    moves = {}
    for _, _, vb, va in cells:
        k = f"{ARC_COLOR_CHARS[max(0,min(15,vb))]}>{ARC_COLOR_CHARS[max(0,min(15,va))]}"
        moves[k] = moves.get(k, 0) + 1
    top = sorted(moves.items(), key=lambda kv: -kv[1])[:3]
    return {"changed": len(cells), "bbox": [min(rows), min(cols), max(rows), max(cols)],
            "moves": " ".join(f"{k}x{v}" for k, v in top)}

ACTS = {i: getattr(GameAction, f"ACTION{i}") for i in range(1, 8)}
LOG = []

def grid_of(o):
    return tuple(tuple(int(v) for v in row) for row in o.frame[-1])

class Game:
    def __init__(self, gid):
        self.arc = arc_agi.Arcade()
        self.env = self.arc.make(gid)
        o = self.env.reset()
        self.g = grid_of(o)
        self.n = 0

    def act(self, a, r=None, c=None):
        before = self.g
        if a == 6:
            o = self.env.step(GameAction.ACTION6, {"x": int(c), "y": int(r)})
            disp = f"MOUSE({r},{c})"
        else:
            o = self.env.step(ACTS[a])
            disp = f"ACTION{a}"
        self.n += 1
        if o is None or getattr(o, "frame", None) is None:
            LOG.append(dict(action=disp, frames=0, changed=0, note="TERMINAL"))
            return
        frames = len(o.frame)
        self.g = grid_of(o)
        d = _describe_change(before, self.g)
        row = dict(action=disp, frames=frames, **d)
        if a == 6:
            row["clicked"] = _cell_color_char(before, r, c)
        LOG.append(row)

    def table(self, last=14):
        print(f"{'#':>3}  {'action':<16}{'frames':>7}{'changed':>8}  {'where':<18}{'what moved':<24}note")
        for i, row in enumerate(LOG[-last:], start=max(1, len(LOG)-last+1)):
            b = row.get("bbox")
            where = f"r{b[0]}-{b[2]} c{b[1]}-{b[3]}" if b else "-"
            note = []
            if row.get("changed", 0) == 0: note.append("board identical")
            if row.get("clicked"): note.append(f"clicked {row['clicked']}")
            if row.get("note"): note.append(row["note"])
            print(f"{i:>3}  {row['action']:<16}{row['frames']:>7}{row.get('changed',0):>8}  "
                  f"{where:<18}{str(row.get('moves') or '-')[:23]:<24}{', '.join(note)}")

    def show(self, r0=0, r1=64, c0=0, c1=64):
        print("    " + "".join(str(c % 10) for c in range(c0, min(c1, 64))))
        for y in range(r0, min(r1, 64)):
            print(f"{y:2d}  " + "".join(ARC_COLOR_CHARS[self.g[y][x]] for x in range(c0, min(c1, 64))))
