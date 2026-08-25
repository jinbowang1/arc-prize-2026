"""我手动玩游戏的小工具: 离线加载, 渲染帧(十六进制字符图, 每格1字符), 连通块表, 克隆体试动作看diff."""
import copy, os, sys
os.environ.setdefault("OPERATION_MODE", "OFFLINE"); os.chdir(os.path.expanduser("~/Desktop/project/arc-agi-3"))
from collections import Counter
import numpy as np, arc_agi
from arcengine import GameAction, ActionInput
ACTS = {i: getattr(GameAction, f"ACTION{i}") for i in range(1, 8)}
_arc = None
def load(gid):
    global _arc
    _arc = _arc or arc_agi.Arcade()
    env = _arc.make(gid)
    o = env.reset()
    return env, o
def grid(o): return np.array(o.frame[-1])
def show(g, r0=0, r1=64, c0=0, c1=64):
    print("    " + "".join(str(c % 10) for c in range(c0, c1)))
    for y in range(r0, r1):
        print(f"{y:2d}  " + "".join(format(int(v), "x") for v in g[y, c0:c1]))
def comps(g, bg=None):
    h, w = g.shape; bg = Counter(g.flatten().tolist()).most_common(1)[0][0] if bg is None else bg
    seen = np.zeros_like(g, bool); out = []
    for y in range(h):
        for x in range(w):
            if seen[y, x] or g[y, x] == bg: continue
            c = g[y, x]; st = [(y, x)]; seen[y, x] = True; cells = []
            while st:
                a, b = st.pop(); cells.append((a, b))
                for na, nb in ((a-1,b),(a+1,b),(a,b-1),(a,b+1)):
                    if 0 <= na < h and 0 <= nb < w and not seen[na, nb] and g[na, nb] == c:
                        seen[na, nb] = True; st.append((na, nb))
            ys, xs = zip(*cells)
            out.append(dict(color=int(c), n=len(cells), box=(min(ys), max(ys), min(xs), max(xs)), center=(int(np.mean(ys)), int(np.mean(xs)))))
    return sorted(out, key=lambda d: -d["n"]), bg
def clone(env):
    game = env._game if hasattr(env, "_game") else env.game
    clean = getattr(game, "_clean_levels", None); game._clean_levels = None
    g2 = copy.deepcopy(game); game._clean_levels = clean; g2._clean_levels = clean
    return g2
def act(env, a, x=None, y=None):
    """在真环境上走一步, 返回 obs"""
    if a == 6: return env.step(ACTS[6], data={"x": int(x), "y": int(y)})
    return env.step(ACTS[a])
def diff(g0, g1):
    d = np.argwhere(g0 != g1)
    if len(d) == 0: return "无变化"
    by = Counter((int(g0[y, x]), int(g1[y, x])) for y, x in d)
    ys, xs = d[:, 0], d[:, 1]
    return f"{len(d)}格变 行{ys.min()}-{ys.max()} 列{xs.min()}-{xs.max()} 色变:{dict(by.most_common(6))}"
