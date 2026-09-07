"""winframe A/B: 按官方计分公式逐局配对对比。

游戏分 = Σ已过关[ 关序i × min(1.15, (基准/该关累计步数)^2) ] / Σ全部关序i
"""
import json, glob, os
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))

def game_score(base_steps, actual_steps, levels_completed, n_levels):
    denom = sum(range(1, n_levels + 1))
    total = 0.0
    for i in range(levels_completed):
        b = base_steps[i] if i < len(base_steps) else 0
        a = actual_steps[i] if i < len(actual_steps) else 0
        if a <= 0:
            eff = 1.15
        else:
            eff = min(1.15, (b / a) ** 2)
        total += (i + 1) * eff
    return 100.0 * total / denom if denom else 0.0

def collect(tag):
    rows = []
    for bj in glob.glob(f"{BASE}/ds_{tag}/*/benchmark.json"):
        try: d = json.load(open(bj, encoding="utf-8"))
        except Exception: continue
        for r in d.get("game_runs", []):
            if not isinstance(r, dict): continue
            # 跳过还在跑的局: benchmark.json 是周期保存的, 未完成的局 state 仍是 playing
            if str(r.get("state", "")).lower() == "playing":
                continue
            lc = int(r.get("levels_completed") or 0)
            ap = r.get("actions_per_level") or []
            bp = r.get("base_actions_per_level") or []
            nl = int(r.get("number_of_levels") or len(bp) or 1)
            rows.append(dict(
                game=str(r.get("game_id", ""))[:4],
                levels=lc,
                actions=sum(int(x or 0) for x in ap),
                score=game_score(bp, ap, lc, nl),
            ))
    return rows

out = {}
for tag in ("wf0", "wf1"):
    rows = collect(tag)
    if not rows:
        print(f"[{tag}] 没有数据"); continue
    per = defaultdict(list)
    for r in rows: per[r["game"]].append(r)
    out[tag] = dict(rows=rows, per=per)
    lv = sum(r["levels"] for r in rows)
    op = sum(1 for r in rows if r["levels"] > 0)
    sc = sum(r["score"] for r in rows) / len(rows)
    ac = sum(r["actions"] for r in rows)
    label = "关(对照)" if tag == "wf0" else "开(winframe)"
    print(f"[{tag} {label}] n={len(rows)}  总过关={lv}  开张={op}/{len(rows)}  均分={sc:.2f}  总步数={ac}")

if "wf0" in out and "wf1" in out:
    a, b = out["wf0"], out["wf1"]
    la = sum(r["levels"] for r in a["rows"]); lb = sum(r["levels"] for r in b["rows"])
    oa = sum(1 for r in a["rows"] if r["levels"] > 0); ob = sum(1 for r in b["rows"] if r["levels"] > 0)
    sa = sum(r["score"] for r in a["rows"])/len(a["rows"]); sb = sum(r["score"] for r in b["rows"])/len(b["rows"])
    print("\n=== winframe 开 减 关 ===")
    print(f"  总过关 : {la} -> {lb}   ({lb-la:+d})")
    print(f"  开张   : {oa} -> {ob}   ({ob-oa:+d})")
    print(f"  均分   : {sa:.2f} -> {sb:.2f}  ({sb-sa:+.2f})")

    # 聚焦: winframe 只在开了张的局里才有意义
    ea = sum(r["levels"]-1 for r in a["rows"] if r["levels"] > 0)
    eb = sum(r["levels"]-1 for r in b["rows"] if r["levels"] > 0)
    print(f"\n  [聚焦] 开张后的额外过关: {ea} -> {eb}  ({eb-ea:+d})   <- winframe 真正作用的量")
    if oa and ob:
        print(f"  [聚焦] 每开张局平均额外过关: {ea/oa:.2f} -> {eb/ob:.2f}")

    print("\n=== 逐局配对 ===")
    for g in sorted(set(a["per"]) | set(b["per"])):
        ra, rb = a["per"].get(g, []), b["per"].get(g, [])
        la_ = sum(r["levels"] for r in ra); lb_ = sum(r["levels"] for r in rb)
        sa_ = sum(r["score"] for r in ra)/len(ra) if ra else 0
        sb_ = sum(r["score"] for r in rb)/len(rb) if rb else 0
        mark = "↑" if lb_ > la_ else ("↓" if lb_ < la_ else "=")
        print(f"  {g}: 关 {la_}->{lb_} {mark}  分 {sa_:.1f}->{sb_:.1f}  (n={len(ra)}/{len(rb)})")
