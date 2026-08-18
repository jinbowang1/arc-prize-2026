"""harness 成绩单: 零介入跑所有游戏, 记录过关数与步数效率。

**这是 harness 的度量, 不是某一局的攻关。** 没有度量就谈不上"越来越强"。

课程式递进(从易到难, 人类基准合计):
    cd82 171 -> ft09 208 -> sb26 213 -> r11l 233 -> tn36 317 -> sc25 350 -> ...
每关限时, 因为比赛的现实预算是**每局几分钟**, 不是几十分钟 ——
靠表征和启发式赢, 不靠枚举。

用法:
    uv run python bench.py                  # 跑默认这批
    uv run python bench.py cd82 sc25        # 只跑指定的
    uv run python bench.py --seconds 30     # 改每关预算
输出 results/bench_<日期>.json, 便于历次对比。
"""
from __future__ import annotations

import glob
import json
import sys
import time

from harness.agent import solve_game

DEFAULT = ["cd82", "ft09", "sc25", "r11l", "ls20", "tr87"]


def main() -> None:
    argv = sys.argv[1:]
    secs = 60.0
    args = []
    i = 0
    while i < len(argv):
        if argv[i] == "--seconds":          # ⚠️它的**值**不能被当成游戏名
            secs = float(argv[i + 1]); i += 2
        elif argv[i].startswith("--"):
            i += 1
        else:
            args.append(argv[i]); i += 1
    games = args or DEFAULT

    cfg = {"bfs_seconds": secs, "best_first_seconds": secs,
           "slot_seconds": secs, "abstract_seconds": secs / 2,
           "layer_seconds": secs / 2, "max_nodes": 20000}
    rows = []
    print(f"=== harness 成绩单 (每关每档 {secs:.0f}s, 零介入) ===\n", flush=True)
    for gid in games:
        try:
            meta = json.load(open(glob.glob(f"environment_files/{gid}/*/metadata.json")[0]))
            base = meta.get("baseline_actions") or []
        except Exception:
            base = []
        t0 = time.time()
        try:
            log = solve_game(gid, baselines=base, cfg=dict(cfg))
            solved = sum(1 for r in log.records if r.solved)
            steps = sum(r.steps for r in log.records if r.solved)
            human = sum(base[:solved]) if base else 0
            rows.append({"game": gid, "solved": solved, "levels": len(base),
                         "steps": steps, "human": human,
                         "seconds": round(time.time() - t0, 1)})
        except Exception as e:
            rows.append({"game": gid, "solved": 0, "levels": len(base),
                         "error": f"{type(e).__name__}: {str(e)[:60]}",
                         "seconds": round(time.time() - t0, 1)})
        r = rows[-1]
        eff = f"{r['steps']}/{r['human']}" if r.get("human") else "-"
        print(f"  {r['game']:6} 过 {r['solved']}/{r['levels']} 关 | 步数 {eff} | "
              f"{r['seconds']:.0f}s {r.get('error','')}", flush=True)

    total_solved = sum(r["solved"] for r in rows)
    total_levels = sum(r["levels"] for r in rows)
    print(f"\n合计: **{total_solved}/{total_levels} 关**, "
          f"总耗时 {sum(r['seconds'] for r in rows):.0f}s", flush=True)
    out = f"results/bench_{time.strftime('%Y%m%d_%H%M')}.json"
    json.dump({"seconds_per_stage": secs, "rows": rows,
               "total_solved": total_solved, "total_levels": total_levels},
              open(out, "w"), ensure_ascii=False, indent=1)
    print(f"已存 {out}", flush=True)


main()
