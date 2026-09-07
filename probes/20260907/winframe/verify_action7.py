"""纯引擎验证 ACTION7 是不是 undo: 走一步 -> ACTION7 -> 盘面是否回到走之前。"""
import os, sys, random
sys.path.insert(0, os.path.expanduser("~/Desktop/project/arc-agi-3/reference/fnrep-source-v4/src/ARC3-Inference"))
sys.path.insert(0, os.path.expanduser("~/Desktop/project/arc-agi-3/reference/fnrep-source-v4/src/tufa-arc-agi-framework/src"))
import numpy as np, arcengine, taaf.game_api

ENV = os.path.expanduser("~/Desktop/project/arc-agi-3/environment_files")
GAME = sys.argv[1]
spec = taaf.game_api.ArcadeSpec(environments_dir=ENV)
g = taaf.game_api.GameAPI(env_name=GAME, arcade_spec=spec); g.start_game()
rng = random.Random(0)

def grid(): return np.asarray(g.current_state.frame.data).copy()
def act(name, data=None):
    return g.execute_action(arcengine.ActionInput(id=arcengine.GameAction.from_name(name), data=data or {}),
                            generated_tokens=0, uncached_input_tokens=0)

avail = [int(a) for a in g.current_state.available_actions]
names = ["RESET" if a == 0 else f"ACTION{a}" for a in avail]
print(f"[{GAME}] 可用动作: {names}")
if 7 not in avail:
    print("  该局没有 ACTION7"); raise SystemExit

tested = restored = 0
for trial in range(12):
    others = [a for a in avail if a not in (0, 7)]
    if not others: break
    a = rng.choice(others)
    nm = f"ACTION{a}"
    data = {"x": rng.randrange(64), "y": rng.randrange(64)} if a == 6 else {}
    before = grid()
    act(nm, data)
    after = grid()
    if np.array_equal(before, after):
        continue          # 这一步没改变盘面, 测不出 undo
    act("ACTION7")
    undone = grid()
    tested += 1
    ch_fwd = int(np.count_nonzero(before != after))
    diff = (undone != before)
    ch_back = int(np.count_nonzero(diff))
    # 排除边缘一圈(HUD 常驻的位置)后再比
    inner = diff[1:-1, 1:-1]
    ch_inner = int(np.count_nonzero(inner))
    where = np.argwhere(diff)
    loc = ",".join(f"({r},{c})" for r, c in where[:4])
    ok = ch_inner == 0
    restored += ok
    print(f"  试{tested}: {nm} 改了{ch_fwd}格 -> 还原后总差{ch_back}格, 去掉边缘一圈差{ch_inner}格  残差位置 {loc}  {'✅盘面已还原' if ok else '❌未还原'}")
    if tested >= 6: break
print(f"\n结论: {restored}/{tested} 次 ACTION7 完全还原了上一步")
