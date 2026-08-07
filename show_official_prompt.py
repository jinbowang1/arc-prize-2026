"""按官方 baseline 的格式, 生成 ls20 第一帧真正发给 LLM 的 prompt, 并算它多大。

格式抄自 reference/ARC-AGI-3-Agents/agents/templates/llm_agents.py 的
build_user_prompt / build_func_resp_prompt / pretty_print_3d。
"""
import textwrap
import numpy as np
import arc_agi

arc = arc_agi.Arcade()
env = arc.make("ls20")
f = env.reset()


def pretty_print_3d(array_3d):
    lines = []
    for i, block in enumerate(array_3d):
        lines.append(f"Grid {i}:")
        for row in block:
            lines.append(f"  {row}")
        lines.append("")
    return "\n".join(lines)


user_prompt = textwrap.dedent("""
# CONTEXT:
You are an agent playing a dynamic game. Your objective is to
WIN and avoid GAME_OVER while minimizing actions.

One action produces one Frame. One Frame is made of one or more sequential
Grids. Each Grid is a matrix size INT<0,63> by INT<0,63> filled with
INT<0,15> values.

# TURN:
Call exactly one action.
""")

obs = f"""
# State:
{f.state.name}

# Score:
{f.levels_completed}

# Frame:
{pretty_print_3d(f.frame)}

# TURN:
Reply with a few sentences of plain-text strategy observation about the frame to inform your next action.
"""

print("=" * 70)
print("【官方 baseline 的全部指令】(这是 LLM 拿到的所有背景知识)")
print("=" * 70)
print(user_prompt)

print("=" * 70)
print("【每一步的观测】前 12 行 + 后 6 行")
print("=" * 70)
lines = obs.strip().splitlines()
for l in lines[:12]:
    print(l[:200])
print(f"  ... 中间省略 {len(lines) - 18} 行, 每行是一整行 64 个整数 ...")
for l in lines[-6:]:
    print(l[:200])

print("\n" + "=" * 70)
print("【体积】")
print("=" * 70)
print(f"  指令部分   {len(user_prompt):>7} 字符")
print(f"  单帧观测   {len(obs):>7} 字符  (约 {len(obs)//4} token)")
print(f"  帧本体行数 {len(f.frame)} 层 x 64 行")
print(f"  → 每走一步就要重发一次这么大的观测。")
print(f"  → 我的解 335 步; 若每步都发, 光观测就是 {len(obs)*335//4//1000}K token 量级。")

g = np.array(f.frame[-1])
print("\n" + "=" * 70)
print("【核对: 官方 GuidedLLM 里那份人工规则, 跟这个环境对得上吗】")
print("=" * 70)
present = {int(v): int(n) for v, n in zip(*np.unique(g, return_counts=True))}
print(f"  本帧实际出现的颜色索引: {present}")
for claim, idx in [("墙 = INT<10>", 10), ("地板 = INT<8>", 8),
                   ("能量补给 = 2x2 的 INT<6>", 6), ("出口门边框 = INT<11>", 11)]:
    n = present.get(idx, 0)
    print(f"  官方规则称 {claim:<28} → 本帧该色像素数 = {n}"
          + ("   ✗ 根本不存在" if n == 0 else ""))
print(f"\n  官方规则称 玩家是 4x4                  → 实测是 5x5(上2行色12 + 下3行色9)")
print(f"  官方规则称 6 levels                    → 实测 win_levels = {f.win_levels}")
