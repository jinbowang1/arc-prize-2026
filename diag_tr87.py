"""逆向 L1: 读过关前一刻的答案区真实符号, 与题面/字典比对找真实规则。"""
import copy, json
import numpy as np
import arc_agi
from arcengine import GameAction, ActionInput
from parse_tr87 import parse, show

ACTS = {i: getattr(GameAction, f"ACTION{i}") for i in range(1, 8)}

arc = arc_agi.Arcade()
env = arc.make("tr87")
f = env.reset()
g_l1 = np.array(f.frame[-1])
pairs1, prob1, ans1_init, _ = parse(g_l1)

l1 = json.load(open("tr87_l1.json"))
seq = l1["l1_seq"]
for a in seq[:-1]:
    f = env.step(ACTS[a])
g_pre = np.array(f.frame[-1])
_, _, ans_pre, _ = parse(g_pre)
f = env.step(ACTS[seq[-1]])
assert f.levels_completed == 1
# 最后一步是 ACTION2(位5 反循环一次) -> 终态答案 = ans_pre 改位5
# 直接用 clone 不行了, 改从倒数第二帧手工推最后一步? 不如重开一局在 clone 上走完整序列拿终帧
import arc_agi as A2
env2 = A2.Arcade().make("tr87")
f2 = env2.reset()
game2 = env2._game
def raw(g, a):
    return g.perform_action(ActionInput(id=ACTS[a], data={}), raw=True)
last_frame = None
for a in seq:
    fr = raw(game2, a)
    if fr.frame:
        last_frame = np.array(fr.frame[-1])
    if fr.levels_completed:
        break
# fr.levels_completed=1 时 last_frame 可能已是 L2 画面; 取倒数第二帧重放到 n-1 再单帧算
env3 = A2.Arcade().make("tr87")
f3 = env3.reset()
game3 = env3._game
for a in seq[:-1]:
    raw(game3, a)
g_last = None
fr = raw(game3, seq[-1])
frames = fr.frame
print(f"最后一步返回 {len(frames)} 帧, levels={fr.levels_completed}")
for i, fim in enumerate(frames):
    gg = np.array(fim)
    try:
        _, _, aa, _ = parse(gg)
        tag = " ".join(show(m)[:11] for m in aa)
        print(f"  帧{i}: 答案区前缀 {tag}")
    except Exception as e:
        print(f"  帧{i}: 解析失败 {e}")

# 用 帧0(过关判定前的最终答案画面) 做 ground truth
g_final = np.array(frames[0])
_, _, ans_final, _ = parse(g_final)

print("\n=== L1 字典 ===")
for s, d in pairs1:
    print("  ", " ".join(show(m) for m in s), " -> ", " ".join(show(m) for m in d))
print("=== L1 题面 ===")
for m in prob1:
    print("  ", show(m))
print("=== L1 正确答案(真实) ===")
for m in ans_final:
    print("  ", show(m))

# 规则挖掘: 答案[i] 与 题面[i] 的像素关系; 和字典对的像素关系
def rel(a, b):
    """求 b 相对 a 的变换: 恒等/转90/180/270/镜像 x 是否取反"""
    a = np.array(a); b = np.array(b)
    ops = {"id": lambda x: x, "r90": lambda x: np.rot90(x, -1), "r180": lambda x: np.rot90(x, 2),
           "r270": lambda x: np.rot90(x, 1), "fliplr": np.fliplr, "flipud": np.flipud,
           "transpose": lambda x: x.T, "anti-transpose": lambda x: np.rot90(x.T, 2)}
    out = []
    for name, op in ops.items():
        if np.array_equal(op(a), b):
            out.append(name)
        if np.array_equal(1 - op(a), b):
            out.append(f"inv-{name}")
    return out

print("\n=== 题面[i] -> 答案[i] 几何关系 ===")
for i, (p, a) in enumerate(zip(prob1, ans_final)):
    print(f"  位{i+1}: {rel(p, a) or '无简单几何关系'}")
print("=== 字典 src -> dst 几何关系 ===")
for s, d in pairs1:
    if len(s) == 1 and len(d) == 1:
        print(f"  {rel(s[0], d[0]) or '无'}")
