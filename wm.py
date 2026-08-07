"""ls20 世界模型: 感知 + 转移表学习 + 模型上规划 + 真机验证。

设计要点
- 不深拷贝: 在真模拟器里连续行走采样转移(~700 步/秒), 学出格级转移表。
- 状态抽象 = (格子, 形状, 关卡); 能量作为资源单独规划, 不进抽象状态。
- 规划在学到的表上做(微秒级), 再回真模拟器逐步验证; 有分歧就当反例修模型。
"""
import json
import numpy as np
import arc_agi
from arcengine import GameAction, ActionInput

ACTS = {a: getattr(GameAction, f"ACTION{a}") for a in (1, 2, 3, 4)}
KEY_TOP, KEY_BOT = 12, 9          # 钥匙块: 上2行色12, 下3行色9
PANEL_R0, PANEL_C0 = 55, 3        # 左下形状面板 6x6, 每 2x2 一个格


def step(game, a):
    return game.perform_action(ActionInput(id=ACTS[a], data={}), raw=True)


def key_candidates(g):
    """所有符合钥匙图案的 5x5 块左上角(锁的显示屏也符合, 故需按连续性甄别)。"""
    out = []
    for r in range(0, 56):
        for c in range(0, 60):
            w = g[r:r + 5, c:c + 5]
            if (w[0:2] == KEY_TOP).all() and (w[2:5] == KEY_BOT).all():
                out.append((r, c))
    return out


def shape_bits(g):
    """左下面板 3x3 形状 -> 9 位整数, 非背景色(5)记 1。"""
    bits = 0
    for i in range(3):
        for j in range(3):
            blk = g[PANEL_R0 + i * 2: PANEL_R0 + i * 2 + 2,
                    PANEL_C0 + j * 2: PANEL_C0 + j * 2 + 2]
            if (blk != 5).any():
                bits |= 1 << (i * 3 + j)
    return bits


def panel_color(g):
    """左下面板的颜色(锁要求形状+颜色都匹配)。"""
    for i in range(3):
        for j in range(3):
            blk = g[PANEL_R0 + i * 2: PANEL_R0 + i * 2 + 2,
                    PANEL_C0 + j * 2: PANEL_C0 + j * 2 + 2]
            v = set(blk.flatten().tolist()) - {5}
            if v:
                return sorted(v)[0]
    return None


def energy(g):
    return int((g[61] == 11).sum())


class Percept:
    """按连续性跟踪钥匙, 避开静态锁显示屏的干扰。"""

    def __init__(self, g):
        self.last = None
        cands = key_candidates(g)
        self.static = set(cands)   # 首帧候选先全记为疑似静态
        self.last = cands[0] if cands else None

    def calibrate(self, g_before, g_after):
        """用一次真实移动区分: 位置变了的才是钥匙。"""
        b, a = set(key_candidates(g_before)), set(key_candidates(g_after))
        moved = a - b
        self.static = b & a
        if moved:
            self.last = sorted(moved)[0]
        return self.last

    def key(self, g):
        cands = [p for p in key_candidates(g) if p not in self.static]
        if not cands:
            return self.last
        if self.last is None:
            self.last = cands[0]
        else:
            self.last = min(cands, key=lambda p: abs(p[0] - self.last[0]) + abs(p[1] - self.last[1]))
        return self.last

    def state(self, g, level):
        """形状维用 (形状bits, 颜色) 元组: L5起过关要求两者同时匹配。"""
        return (self.key(g), (shape_bits(g), panel_color(g)), level)


def load_env(prefix_file="solutions.json", game_id="ls20"):
    prefix = json.load(open(prefix_file))["seq"]
    arc = arc_agi.Arcade()
    env = arc.make(game_id)
    env.reset()
    game = env._game
    f = None
    for a in prefix:
        f = step(game, a)
    return game, f


if __name__ == "__main__":
    game, f = load_env()
    g = np.array(f.frame[-1])
    p = Percept(g)
    print("candidates at L5 start:", key_candidates(g))
    f2 = step(game, 1)
    g2 = np.array(f2.frame[-1])
    print("after a1 candidates:", key_candidates(g2))
    print("calibrated key:", p.calibrate(g, g2))
    print("state:", p.state(g2, f2.levels_completed),
          "energy:", energy(g2))
