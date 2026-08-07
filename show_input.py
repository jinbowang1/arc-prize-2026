"""把"AI 实际看到的输入"原样打出来: 帧的数据类型/形状/取值, 以及一帧的完整内容。"""
import numpy as np
import arc_agi
from arcengine import GameAction

arc = arc_agi.Arcade()
env = arc.make("ls20")
f = env.reset()

print("=== reset() 返回的对象有哪些字段 ===")
for k in sorted(f.model_dump().keys()):
    v = getattr(f, k)
    if k == "frame":
        print(f"  {k:<20} list[{len(v)}] 层, 每层 {len(v[0])}x{len(v[0][0])} 整数")
    else:
        print(f"  {k:<20} {v!r}")

g = np.array(f.frame[-1])
print(f"\n=== 帧本体 ===")
print(f"  形状 {g.shape}  类型 {g.dtype}  取值范围 {g.min()}~{g.max()}")
print(f"  出现的颜色索引及像素数: {dict(zip(*[x.tolist() for x in np.unique(g, return_counts=True)]))}")

print(f"\n=== 这就是 AI 看到的全部(L1 开局, 64x64 颜色索引) ===")
print("     " + "".join(str(c % 10) for c in range(64)))
CH = ".123456789ABCDEF"
for r in range(64):
    print(f"{r:>3}  " + "".join(CH[v] for v in g[r]))

print("\n=== 动作空间 ===")
print(f"  {[a.name for a in GameAction]}")
print("  没有任何文字说明: 动作叫什么、按下会发生什么、目标是什么, 全都要自己试出来")

print("\n=== 走一步之后, 反馈是什么 ===")
f2 = env.step(GameAction.ACTION1)
g2 = np.array(f2.frame[-1])
d = np.argwhere(g != g2)
print(f"  变化像素 {len(d)} 个, 分布在行 {sorted({int(r) for r,_ in d})}")
print(f"  state={f2.state.name} levels_completed={f2.levels_completed} "
      f"available_actions={f2.available_actions}")
print("  注意: 连 score 字段都没有——没有奖励信号、没有提示、没有'你离目标还有多远'。")
print("  整局唯一的外部反馈就是 levels_completed 这个计数器, 通关一关它 +1。")
