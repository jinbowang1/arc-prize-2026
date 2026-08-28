"""零成本重放: 统计 25 个公开局的动画分布(type1 信息藏中间帧 / type2 纯插值)。

口径直接用 LB-9 源码包里 Jakob 的 summarize_animation, 保证和他 24 局的实测可比。
不跑模型, 只用随机+系统探测走动作, 记录每个动作返回几帧、藏了多少 transient pixel。
"""
import os, sys, json, random, argparse
os.environ.setdefault("OPERATION_MODE", "OFFLINE")
os.chdir(os.path.expanduser("~/Desktop/project/arc-agi-3"))
sys.path.insert(0, "/private/tmp/claude-1518879226/-Users-01450825/73b29d8b-5d3c-4369-b798-7cbab154e4ee/scratchpad/lb9src/src/ARC3-Inference")
import numpy as np, arc_agi
from arcengine import GameAction
from inference.utils.animation import summarize_animation, normalize_frames

ACTS = {i: getattr(GameAction, f"ACTION{i}") for i in range(1, 8)}

def step(env, a, x=None, y=None):
    if x is None:
        return env.step(ACTS[a])
    return env.step(GameAction.ACTION6, {"x": int(x), "y": int(y)})

def scan_game(gid, steps, seed):
    rng = random.Random(seed)
    arc = arc_agi.Arcade(); env = arc.make(gid); o = env.reset()
    prev = normalize_frames([o.frame[-1]])[0]
    recs = []; resets = 0
    # 前 7 步: 每类动作各试一遍(Jakob 的 bootstrap probe 口径)
    plan = [(i, None, None) for i in range(1, 6)] + [(7, None, None)]
    for n in range(steps):
        if n < len(plan):
            a, x, y = plan[n]
        elif rng.random() < 0.55:
            a, x, y = 6, rng.randrange(64), rng.randrange(64)
        else:
            a, x, y = rng.choice([1, 2, 3, 4, 5, 7]), None, None
        try:
            o = step(env, a, x, y)
        except Exception:
            continue
        if o is None or getattr(o, "frame", None) is None:
            # 游戏进入终局(GAME_OVER/WIN): 重开一局继续采样, 不然覆盖不足
            o = env.reset(); prev = normalize_frames([o.frame[-1]])[0]; resets += 1
            continue
        frames = normalize_frames(o.frame)
        if not frames:
            continue
        cur = frames[-1]
        changed = cur != prev
        s = summarize_animation(frames, board_changed=changed)
        recs.append({
            "action": f"MOUSE({y},{x})" if a == 6 else f"ACTION{a}",
            "frames": len(frames), "board_changed": changed,
            "transient": (s or {}).get("transient_pixels", 0),
            "board_unchanged": bool((s or {}).get("board_unchanged")),
        })
        prev = cur
        if getattr(o, "state", None) is not None and str(o.state).endswith("WIN"):
            break
    return recs, resets

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--games", default="")
    ap.add_argument("--out", default="probes/20260827/anim_scan.json")
    args = ap.parse_args()
    games = args.games.split(",") if args.games else sorted(g for g in os.listdir("environment_files") if not g.startswith(".") and g != "dc22")
    allrec = {}
    for i, g in enumerate(games, 1):
        recs, resets = scan_game(g, args.steps, args.seed)
        multi = [r for r in recs if r["frames"] > 1]
        tr = [r for r in multi if r["transient"] > 0]
        hidden = [r for r in multi if r["board_unchanged"] and r["transient"] > 0]
        allrec[g] = recs
        print(f"[{i:2}/{len(games)}] {g}: 动作{len(recs):4} 多帧{len(multi):4}({100*len(multi)/max(1,len(recs)):4.1f}%) "
              f"带藏信息{len(tr):4} 其中棋盘没变{len(hidden):4} "
              f"最大帧数{max([r['frames'] for r in recs], default=0):3} 重开{resets}", flush=True)
        json.dump(allrec, open(args.out, "w"))
    print("saved", args.out)
