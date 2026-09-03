# -*- coding: utf-8 -*-
"""第一层 A/B 的第一关视角报表 (08-31 第一关策略专用)。

判据(与旧 ab_report.py 的机制指标互补): 第一关通过率 / 第一关步数与效率(平方项) /
按官方公式折算的每局分。用法: l1_report.py <run目录|run名> [<run目录2> ...]
run名相对 probes/20260831/aiplat_ab 解析。
"""
import json, os, sys, glob, statistics

ROOT = os.path.expanduser("~/Desktop/project/arc-agi-3/probes/20260831/aiplat_ab")


def load(run):
    d = run if os.path.isdir(run) else os.path.join(ROOT, run)
    hits = glob.glob(os.path.join(d, "benchmark.json")) or glob.glob(os.path.join(d, "*", "benchmark.json"))
    if not hits:
        raise SystemExit(f"找不到 benchmark.json: {d}")
    return json.load(open(hits[0])), os.path.basename(d.rstrip("/"))


def official_score(base, actions, completed):
    total_w = sum(i + 1 for i in range(len(base)))
    s = 0.0
    for i, b in enumerate(base):
        if i < completed and i < len(actions) and actions[i] > 0:
            s += min(115.0, (b / actions[i]) ** 2 * 100) * (i + 1)
    return s / total_w if total_w else 0.0


def report(run):
    bench, name = load(run)
    rows = []
    for gr in bench["game_runs"]:
        base = gr.get("base_actions_per_level") or []
        acts = gr.get("actions_per_level") or []
        done = int(gr.get("levels_completed") or 0)
        gid = str(gr.get("game_id", "?"))[:4]
        l1 = done >= 1
        l1_eff = (base[0] / acts[0]) ** 2 if l1 and base and acts and acts[0] else None
        rows.append(dict(gid=gid, done=done, l1=l1,
                         l1_steps=acts[0] if acts else 0, l1_base=base[0] if base else 0,
                         l1_eff=l1_eff, score=official_score(base, acts, done)))
    n = len(rows)
    passed = [r for r in rows if r["l1"]]
    print(f"\n== {name}  ({n} runs) ==")
    print(f"  第一关通过率: {len(passed)}/{n} = {100*len(passed)/max(1,n):.0f}%")
    if passed:
        print(f"  第一关步数(过关局): 中位 {statistics.median(r['l1_steps'] for r in passed):.0f}"
              f" (基准中位 {statistics.median(r['l1_base'] for r in passed):.0f})"
              f" | 效率(基/实)² 中位 {statistics.median(r['l1_eff'] for r in passed):.2f}")
    unpassed = [r for r in rows if not r["l1"]]
    if unpassed:
        print(f"  没过第一关的局: {' '.join(r['gid'] for r in unpassed)} (烧步中位 {statistics.median(r['l1_steps'] for r in unpassed):.0f})")
    print(f"  官方折算分: 均值 {statistics.mean(r['score'] for r in rows):.2f} | 总关数 {sum(r['done'] for r in rows)}")
    print("  逐局: " + "  ".join(f"{r['gid']}:{'过' if r['l1'] else '×'}{r['l1_steps']}步" for r in rows))


for arg in sys.argv[1:]:
    report(arg)
