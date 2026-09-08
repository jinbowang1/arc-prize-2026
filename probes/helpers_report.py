#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""预置解析库 A/B 报告 —— 先验"模型用没用", 再看"有没有效果"。

09-08 的教训: 工作台那场 A/B 测出 1.13 vs 0.65~1.81, 事后才发现 24 局里 19 局
的模型从没定义过一个可复用函数 —— 测的根本不是工作台。所以这次先查使用率:
使用率为零, 效果数字就不必解读。

用: python3 probes/helpers_report.py
"""
import glob
import re
import statistics as st
import sys

BASE = "/Users/01450825/Desktop/project/arc-agi-3/probes/20260908/helpers"
HELPERS = ["grid", "at", "find", "counts", "diff", "crop", "objects", "moved"]


def finished(tag):
    p = "%s/%s.full.log" % (BASE, tag)
    try:
        t = open(p, encoding="utf-8", errors="ignore").read()
    except FileNotFoundError:
        return {}
    return {m[0]: {"lvl": int(m[1]), "score": float(m[2]), "act": int(m[3]), "tok": int(m[4])}
            for m in re.findall(
                r"\[finished\] (\S+?)-\S+ state=\S+ level=(\d+)/\d+ score=([\d.]+) "
                r"actions=(\d+) tokens=(\d+)", t)}


def usage(tag):
    """模型在 python 代码里真的调用了几次预置 helper。"""
    calls, hits, games_using = 0, {h: 0 for h in HELPERS}, 0
    for f in glob.glob("%s/%s/*/transcripts/*.txt" % (BASE, tag)):
        codes = re.findall(r"<parameter=code>(.*?)</parameter>",
                           open(f, encoding="utf-8", errors="ignore").read(), re.S)
        calls += len(codes)
        used_here = False
        for c in codes:
            for h in HELPERS:
                n = len(re.findall(r"(?<![\w.])%s\s*\(" % h, c))
                if n and ("def %s" % h) not in c:
                    hits[h] += n
                    used_here = True
        if used_here:
            games_using += 1
    return calls, hits, games_using


def main():
    a, b = finished("h0"), finished("h1")
    if not a or not b:
        sys.exit("还没跑完: h0 %d 局 / h1 %d 局" % (len(a), len(b)))
    both = sorted(set(a) & set(b))

    print("=== 第一关: 模型到底用没用预置库 ===")
    for tag in ("h0", "h1"):
        calls, hits, games = usage(tag)
        total = sum(hits.values())
        print("  %s: %d 次 python 调用 | helper 被调用 %d 次 | 用过它的局 %d"
              % (tag, calls, total, games))
        if total:
            print("       分布: %s" % ", ".join("%s×%d" % (k, v) for k, v in
                                                sorted(hits.items(), key=lambda x: -x[1]) if v))
    print("  (h0 关着预置库, 应当为 0; h1 若也接近 0, 下面的效果数字就不必解读)")

    print("\n=== 第二关: 效果 ===")
    print("  %-14s %-8s %-10s %-8s %s" % ("组", "均分", "拿分局数", "动作", "输出tok"))
    for nm, d in (("h0 无预置库", a), ("h1 有预置库", b)):
        v = [d[g]["score"] for g in both]
        won = sum(1 for g in both if d[g]["score"] > 0)
        print("  %-14s %-8.2f %-10s %-8.1f %.0f"
              % (nm, st.mean(v), "%d/%d" % (won, len(both)),
                 sum(d[g]["act"] for g in both) / len(both),
                 sum(d[g]["tok"] for g in both) / len(both)))

    w = sum(1 for g in both if b[g]["score"] > a[g]["score"])
    l = sum(1 for g in both if b[g]["score"] < a[g]["score"])
    print("\n  配对 %d 局: 预置库更好 %d, 无预置库更好 %d, 平 %d" % (len(both), w, l, len(both) - w - l))
    wo = sum(1 for g in both if b[g]["score"] > 0 and a[g]["score"] == 0)
    lo = sum(1 for g in both if b[g]["score"] == 0 and a[g]["score"] > 0)
    print("  按拿分局数配对(方差更小的判据): 预置库独赢 %d, 对照独赢 %d" % (wo, lo))
    n = wo + lo
    if n:
        from math import comb
        p = sum(comb(n, k) for k in range(max(wo, lo), n + 1)) / 2 ** n * 2
        print("  符号检验 双侧 p ≈ %.3f %s" % (p, "(显著)" if p < 0.05 else "(不显著, 样本仍不足)"))


if __name__ == "__main__":
    main()
