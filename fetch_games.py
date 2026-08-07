"""下载若干没打过的公开游戏, 报告它们的元信息(动作空间类型/关卡数/人类基准)。

游戏 ID 清单来自 Milestone 1 冠军 duck-harness 的 competition_arcade.py(25 个公开官方环境)。
"""
import json, sys, time
from pathlib import Path
import arc_agi

WANT = sys.argv[1:] or ["tn36", "lf52", "cn04"]

for gid in WANT:
    t0 = time.time()
    try:
        arc = arc_agi.Arcade()          # NORMAL 模式: 会下载到 environment_files/
        env = arc.make(gid)
        if env is None:
            print(f"{gid}: make 返回 None"); continue
        f = env.reset()
        meta = list(Path("environment_files").glob(f"{gid}/*/metadata.json"))
        m = json.load(open(meta[0])) if meta else {}
        print(f"{gid}: tags={m.get('tags')} 关卡={f.win_levels} "
              f"人类基准={m.get('baseline_actions')} "
              f"可用动作={f.available_actions} ({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        print(f"{gid}: 失败 {type(e).__name__}: {e}", flush=True)
