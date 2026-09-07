"""从初始帧自动检测逻辑格点。

判据: 真正的栅格线让"跨界处的行/列差异"显著高于全局平均。
对每个候选周期 p 和相位 off, 比较 (p 的倍数位置的边界强度) / (全局平均边界强度)。
纯算法, 不依赖模型。
"""
import os, sys
sys.path.insert(0, os.path.expanduser("~/Desktop/project/arc-agi-3/reference/fnrep-source-v4/src/ARC3-Inference"))
sys.path.insert(0, os.path.expanduser("~/Desktop/project/arc-agi-3/reference/fnrep-source-v4/src/tufa-arc-agi-framework/src"))
import numpy as np, taaf.game_api

def edge_profile(board, axis):
    """相邻行(或列)之间的差异比例, 长度 n-1。"""
    if axis == 0:
        return np.array([np.mean(board[i] != board[i + 1]) for i in range(board.shape[0] - 1)])
    return np.array([np.mean(board[:, i] != board[:, i + 1]) for i in range(board.shape[1] - 1)])

def detect(prof, max_p=32, min_p=2):
    """返回 (period, phase, 对比度)。对比度 = 栅格线处平均强度 / 非栅格线处平均强度。"""
    n = len(prof)
    mean_all = prof.mean() + 1e-9
    best = (None, None, 0.0)
    for p in range(min_p, min(max_p, n // 2) + 1):
        for off in range(p):
            idx = np.arange(off, n, p)
            if len(idx) < 3:
                continue
            on = prof[idx].mean()
            mask = np.ones(n, bool); mask[idx] = False
            offv = prof[mask].mean() + 1e-9
            contrast = on / offv
            # 要求栅格线本身够强, 且条数合理
            if on > mean_all and contrast > best[2]:
                best = (p, off, contrast)
    return best

ENV = os.path.expanduser("~/Desktop/project/arc-agi-3/environment_files")
for game in sys.argv[1:]:
    try:
        spec = taaf.game_api.ArcadeSpec(environments_dir=ENV)
        g = taaf.game_api.GameAPI(env_name=game, arcade_spec=spec); g.start_game()
        b = np.asarray(g.current_state.frame.data)
    except Exception as e:
        print(f"[{game}] 加载失败: {e}"); continue
    pr, orr, cr = detect(edge_profile(b, 0))   # 行方向(纵向格点)
    pc, oc, cc = detect(edge_profile(b, 1))    # 列方向(横向格点)
    print(f"[{game}] 颜色数={len(np.unique(b))}")
    print(f"   纵向: 周期={pr} 相位={orr} 对比度={cr:.1f}   -> 行栅格线在 y={orr},{(orr or 0)+(pr or 0)},...")
    print(f"   横向: 周期={pc} 相位={oc} 对比度={cc:.1f}   -> 列栅格线在 x={oc},{(oc or 0)+(pc or 0)},...")
