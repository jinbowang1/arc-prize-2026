"""验证 _solved_frame_grid 真的返回「赢的那一刻」而不是新关开局。"""
import os, sys, random, itertools
sys.path.insert(0, os.path.expanduser("~/Desktop/project/arc-agi-3/reference/fnrep-source-v4/src/ARC3-Inference"))
sys.path.insert(0, os.path.expanduser("~/Desktop/project/arc-agi-3/reference/fnrep-source-v4/src/tufa-arc-agi-framework/src"))
import numpy as np, arcengine, taaf.game_api
from inference.framework.solver import _solved_frame_grid, _grid_from_state

ENV_DIR = os.path.expanduser("~/Desktop/project/arc-agi-3/environment_files")
GAME, SEED = sys.argv[1], int(sys.argv[2])
spec = taaf.game_api.ArcadeSpec(environments_dir=ENV_DIR)
game = taaf.game_api.GameAPI(env_name=GAME, arcade_spec=spec); game.start_game()
rng = random.Random(SEED)
RESET = arcengine.ActionInput(id=arcengine.GameAction.from_name("RESET"), data={})

for step in itertools.count(1):
    if step > 8000: print("没过关"); break
    if game.current_state.raw.state == arcengine.GameState.GAME_OVER:
        game.execute_action(RESET, generated_tokens=0, uncached_input_tokens=0); continue
    avail = [a for a in game.current_state.available_actions if a != 0] or list(game.current_state.available_actions)
    aid = int(rng.choice(avail))
    ga = arcengine.GameAction.from_name("RESET" if aid == 0 else f"ACTION{aid}")
    data = {"x": rng.randrange(64), "y": rng.randrange(64)} if ga.name == "ACTION6" else {}
    before = game.current_state.levels_completed
    st = game.execute_action(arcengine.ActionInput(id=ga, data=data), generated_tokens=0, uncached_input_tokens=0)
    if st.levels_completed > before:
        solved = _solved_frame_grid(st)
        shown = _grid_from_state(st)
        raw2 = tuple(tuple(int(c) for c in r) for r in np.asarray(st.raw.frame[-2]).tolist())
        print(f"[{GAME} seed={SEED}] 第 {step} 步过关")
        print(f"  _solved_frame_grid 有返回        : {solved is not None}")
        print(f"  == raw.frame[-2] (赢的那一刻)    : {solved == raw2}")
        print(f"  != 模型原本看到的新关开局        : {solved != shown}")
        diff = sum(1 for a,b in zip(sum(solved,()), sum(shown,())) if a!=b)
        print(f"  两者差异格子                     : {diff}/4096")
        break
    # 非过关步必须返回 None, 否则会往历史里灌垃圾
    if _solved_frame_grid(st) is not None:
        print(f"❌ 第 {step} 步没过关却返回了帧 —— 误触发!"); break
