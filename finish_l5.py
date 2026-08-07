"""L5 收官 v2: 拿到 371 后全程禁行变形器(10,19), 保住形状再去旋转带凑 413。

上一版失败根因: 分段导航只要求"走到某格", 没要求沿途保形状,
最短路恰好穿过变形器把 371 改回了别的族。
"""
import json
import numpy as np
import solve_l5
from solve_l5 import build, plan
from wm import Percept, energy, load_env, shape_bits, step
from validate_lock import find_locks, show

TARGET, XFORMER = 413, (10, 19)
move, xform = build()


def nav(game, f, p, frm, sh, e, to, forbid=()):
    solve_l5.LOCK = to
    r = plan(move, xform, frm, sh, e, None, forbid=forbid)
    if r is None:
        return None
    for a in r[0]:
        f = step(game, a)
        if not f.frame:
            return None
    g = np.array(f.frame[-1])
    return r[0], f, p.key(g), shape_bits(g), energy(g)


def main():
    game, f = load_env("solutions_l4.json")
    base = f.levels_completed
    p = Percept(np.array(f.frame[-1]))
    seq, cell, sh, e = [], (40, 49), 410, 42

    r = nav(game, f, p, cell, sh, e, (10, 14))
    path, f, cell, sh, e = r
    seq += path
    print(f"到变形器旁 {len(path)}步 shape={sh} e={e}")

    for _ in range(8):                      # 进出变形器直到拿到 371
        f = step(game, 4); seq.append(4)
        g = np.array(f.frame[-1]); sh = shape_bits(g); cell = p.key(g); e = energy(g)
        if sh == 371:
            break
        f = step(game, 3); seq.append(3)
        g = np.array(f.frame[-1]); sh = shape_bits(g); cell = p.key(g); e = energy(g)
    print(f"拿到{sh} @ {cell} e={e} 用了{len(seq)}步")
    if sh != 371:
        print("没拿到371"); return

    # 此后全程禁行变形器
    F = (XFORMER,)
    r = nav(game, f, p, cell, sh, e, (10, 9), forbid=F)     # 补给
    if r is None:
        print("到不了补给点(绕开变形器)"); return
    path, f, cell, sh, e = r
    seq += path
    print(f"补给后 cell={cell} shape={sh} e={e}")

    r = nav(game, f, p, cell, sh, e, (35, 19), forbid=F)    # 去旋转带
    if r is None:
        print("到不了旋转带(绕开变形器)"); return
    path, f, cell, sh, e = r
    seq += path
    print(f"到旋转带 cell={cell} shape={sh} e={e}")

    for i in range(16):                     # 真机贪心凑 413
        a = 3 if i % 2 == 0 else 4
        f = step(game, a); seq.append(a)
        if not f.frame:
            print("死亡"); return
        g = np.array(f.frame[-1]); sh = shape_bits(g); cell = p.key(g); e = energy(g)
        print(f"  摇摆{i} shape={sh} cell={cell} e={e}")
        if sh == TARGET:
            print("*** 拿到 413!"); break
    if sh != TARGET:
        print("旋转带没凑出 413"); return

    for tgt in [(5, 44), (10, 54)]:         # 补给后去锁前
        r = nav(game, f, p, cell, sh, e, tgt, forbid=F)
        if r is None:
            print(f"到不了 {tgt}"); return
        path, f, cell, sh, e = r
        seq += path
        print(f"到 {tgt}: shape={sh} e={e}")

    f = step(game, 2)                      # 退一格, 钥匙不再压住显示屏
    g = np.array(f.frame[-1])
    print(f"退后读锁: 位置={p.key(g)} 我方形状={shape_bits(g)}={show(shape_bits(g))}")
    for (r, c, col, bits) in find_locks(g):
        print(f"  锁当前要求@({r},{c}): {bits}={show(bits)}")
    f = step(game, 1)                      # 回到锁前
    g = np.array(f.frame[-1])
    print(f"回到锁前: 位置={p.key(g)} 形状={shape_bits(g)}")
    for a in (1, 2, 3, 4):
        import copy
        e0 = energy(g)
        ff = step(game, a)
        gg = np.array(ff.frame[-1]) if ff.frame else None
        print(f"  试方向{a}: lv={ff.levels_completed} 位置={p.key(gg) if gg is not None else 'X'} "
              f"能量变化={energy(gg)-e0 if gg is not None else 'X'}")
        if ff.levels_completed > base:
            print("  *** 这个方向开了锁!")
            break
        # 走回去
        back = {1: 2, 2: 1, 3: 4, 4: 3}[a]
        ff = step(game, back)
        g = np.array(ff.frame[-1]) if ff.frame else g
    f = step(game, 1); seq.append(1)
    if f.frame and f.levels_completed > base:
        print(f"*** L5 通关! 本关 {len(seq)} 步")
        json.dump({"level5_seq": seq}, open("l5_seq.json", "w"))
    else:
        print("锁未开, shape=", shape_bits(np.array(f.frame[-1])) if f.frame else "死亡")


if __name__ == "__main__":
    main()
