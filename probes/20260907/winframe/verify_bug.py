"""纯引擎验证 winframe bug: 过关瞬间 GameState.frame 拿到的是不是下一关的开局盘。

不需要模型。随机走到 just_won_level, 比对 raw.frame[-1] (模型看到的) 与 raw.frame[-2] (赢的那一刻)。
"""
import os, sys, random, itertools
sys.path.insert(0, os.path.expanduser("~/Desktop/project/arc-agi-3/reference/fnrep-source-v4/src/ARC3-Inference"))
sys.path.insert(0, os.path.expanduser("~/Desktop/project/arc-agi-3/reference/fnrep-source-v4/src/tufa-arc-agi-framework/src"))
import numpy as np, arcengine, taaf.game_api

ENV_DIR = os.path.expanduser("~/Desktop/project/arc-agi-3/environment_files")
GAME = sys.argv[1] if len(sys.argv) > 1 else "vc33"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
MAX_ACTIONS = int(sys.argv[3]) if len(sys.argv) > 3 else 4000

spec = taaf.game_api.ArcadeSpec(environments_dir=ENV_DIR)
game = taaf.game_api.GameAPI(env_name=GAME, arcade_spec=spec)
rng = random.Random(SEED)
game.start_game()
print(f"[{GAME}] seed={SEED} 关数={game.number_of_levels} 开跑, 最多 {MAX_ACTIONS} 步", flush=True)

for step in itertools.count(1):
    if step > MAX_ACTIONS:
        print(f"跑满 {MAX_ACTIONS} 步没过关 —— 换 seed 或换局再试", flush=True); break
    if game.current_state.raw.state == arcengine.GameState.GAME_OVER:
        game.execute_action(arcengine.ActionInput(id=arcengine.GameAction.from_name("RESET"), data={}), generated_tokens=0, uncached_input_tokens=0)
        continue
    avail = [a for a in game.current_state.available_actions if a != 0]  # 不随机 RESET
    if not avail:
        avail = list(game.current_state.available_actions)
    aid = int(rng.choice(avail))
    ga = arcengine.GameAction.from_name("RESET" if aid == 0 else f"ACTION{aid}")
    data = {"x": rng.randrange(64), "y": rng.randrange(64)} if ga.name == "ACTION6" else {}
    before_level = game.current_state.levels_completed
    st = game.execute_action(arcengine.ActionInput(id=ga, data=data), generated_tokens=0, uncached_input_tokens=0)
    if step % 250 == 0:
        print(f"  ..{step} 步, 已过 {st.levels_completed} 关", flush=True)
    if st.levels_completed > before_level:
        frames = st.raw.frame
        print(f"\n🎯 第 {step} 步过关! raw.frame 帧数 = {len(frames)}", flush=True)
        if len(frames) < 2:
            print("只有 1 帧 —— 这一局走的是引擎的绿帧安全网"); break
        last = np.asarray(frames[-1]); prev = np.asarray(frames[-2])
        gs_frame = np.asarray(st.frame.data)
        print(f"GameState.frame (模型看到的) == raw.frame[-1] ? {np.array_equal(gs_frame,last)}")
        print(f"raw.frame[-2] (赢的那一刻) 与 [-1] 是否不同 ? {not np.array_equal(prev,last)}")
        print(f"差异格子数: {int(np.count_nonzero(prev!=last))} / {prev.size}")
        print(f"[-2] 颜色分布: {dict(zip(*[x.tolist() for x in np.unique(prev,return_counts=True)]))}")
        print(f"[-1] 颜色分布: {dict(zip(*[x.tolist() for x in np.unique(last,return_counts=True)]))}")
        break
