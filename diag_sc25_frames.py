"""sc25 第三问: 第一个动作改 0 格 —— 是不是拿到了"动作还没完成"的那一帧?

已知(diag_sc25_alive.py): 同一动作连发, 第 1 步改 0 格, 第 2 步起才有反应。
harness 每一层判"这个动作有没有效果"都是**单步 peek**, 于是全判成无效果 ->
指纹全等 -> BFS 深度 0 报"穷尽" -> 裸跑 1 秒结束。

嫌疑: `env.Game.act` 用的是 `perform_action(..., raw=True)`, 拿到的是动作
**尚未完成**的帧。SDK 上有 `is_action_complete` / `complete_action`,
Tycho 论文里区分"决策帧 vs 动画帧"说的多半就是这件事。

三问:
    ① raw=False 时单步能不能看见效果
    ② is_action_complete 在动作后是 True 还是 False; 手动 complete_action 会怎样
    ③ 若真是滞后一拍: 走 [A1] 再补一帧, 是否等于连走 [A1, A1] 的第一帧
"""
from __future__ import annotations

import numpy as np

import arc_agi
from arcengine import ActionInput
from harness.env import _ACTION_BY_ID, Action, Game

GID = "sc25"
game, obs = Game.make(GID)
g0 = np.array(obs.grid)

print("[① raw=True vs raw=False, 单步]", flush=True)
for raw in (True, False):
    for aid in (1, 3):
        node = game.fork()
        fr = node._g.perform_action(ActionInput(id=_ACTION_BY_ID[aid], data={}), raw=raw)
        g = np.array(fr.frame[-1] if isinstance(fr.frame, list) else fr.frame)
        print(f"    raw={raw!s:<5} A{aid}: 改 {int((g != g0).sum()):>4} 格", flush=True)

print("\n[② is_action_complete / complete_action]", flush=True)
node = game.fork()
gm = node._g
before = np.array(obs.grid)
fr = gm.perform_action(ActionInput(id=_ACTION_BY_ID[1], data={}), raw=True)
g1 = np.array(fr.frame[-1] if isinstance(fr.frame, list) else fr.frame)
print(f"    perform_action 后: 改 {int((g1 != before).sum())} 格 | "
      f"is_action_complete={gm.is_action_complete()}", flush=True)
# 再 perform 一次同样的动作, 看上一次的效果是不是这时候才出现
fr2 = gm.perform_action(ActionInput(id=_ACTION_BY_ID[1], data={}), raw=True)
g2 = np.array(fr2.frame[-1] if isinstance(fr2.frame, list) else fr2.frame)
print(f"    再 perform 一次后: 相对开局改 {int((g2 != before).sum())} 格 "
      f"(若滞后一拍, 这里出现的就是**第一次** A1 的效果)", flush=True)

print("\n[③ 滞后一拍? 单动作+补帧 vs 连走两次的中间帧]", flush=True)
a = Action.key(1)
n1 = game.fork(); o1 = n1.act(a)              # 只走一次
n2 = game.fork(); n2.act(a); o2 = n2.act(a)   # 走两次
print(f"    走 1 次: 改 {int((np.array(o1.grid) != g0).sum())} 格", flush=True)
print(f"    走 2 次: 改 {int((np.array(o2.grid) != g0).sum())} 格", flush=True)

# 同一个动作走 1..6 次, 累计改了多少 —— 看效果是"每次都动"还是"隔一拍"
print("\n[④ 走 N 次 A3 的累计变化(A3 在 alive 诊断里反应最大)]", flush=True)
for n in range(1, 7):
    nd = game.fork()
    o = None
    for _ in range(n):
        o = nd.act(Action.key(3))
    print(f"    走 {n} 次: 相对开局改 {int((np.array(o.grid) != g0).sum()):>4} 格 "
          f"| level={o.level} dead={o.dead}", flush=True)
