#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""公开集 A/B 对照报告 —— 解析两组运行日志的 [finished] 行, 出配对结果。

分数与关卡数只在日志的 [finished] 行里(benchmark.json 不带), 所以吃日志。
用: python3 probes/ab_report.py probes/20260908/workbench/g0.full.log <g1.full.log>
"""
import glob
import json
import sys
from collections import Counter


def load(path):
    import os
    import re
    txt = io_read(path)
    rows = re.findall(
        r"\[finished\] (\S+?)-\S+ state=(\S+) level=(\d+)/(\d+) score=([\d.]+) "
        r"actions=(\d+) tokens=(\d+)", txt)
    if not rows:
        sys.exit("日志里没有 [finished] 行(还没跑完?): " + path)
    out = {}
    for g, st, lv, tot, sc, act, tok in rows:
        out[g] = {"state": st, "levels": int(lv), "of": int(tot),
                  "score": float(sc), "actions": int(act), "tokens": int(tok)}
    return out, os.path.basename(path).replace(".full.log", "")


def io_read(p):
    with open(p, encoding="utf-8", errors="ignore") as f:
        return f.read()


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    a, la = load(sys.argv[1])
    b, lb = load(sys.argv[2])
    both = sorted(set(a) & set(b))
    if not both:
        sys.exit("两组没有共同的局")

    def agg(d, keys, f):
        return sum(f(d[k]) for k in keys) / len(keys)

    print("%-22s %-12s %-12s" % ("", la, lb))
    for name, f, fmt in (
        ("平均分", lambda x: x["score"], "%.2f"),
        ("每局过关层数", lambda x: x["levels"], "%.2f"),
        ("每局动作", lambda x: x["actions"], "%.1f"),
        ("每局输出 tok", lambda x: x["tokens"], "%.0f"),
            ):
        va, vb = agg(a, both, f), agg(b, both, f)
        mark = "  ←" if abs(vb - va) > 1e-9 and vb > va else ""
        print("%-22s %-12s %-12s%s" % (name, fmt % va, fmt % vb, mark))

    w = sum(1 for g in both if b[g]["score"] > a[g]["score"])
    l = sum(1 for g in both if b[g]["score"] < a[g]["score"])
    print("\n配对 %d 局: 后者更好 %d, 前者更好 %d, 打平 %d" % (len(both), w, l, len(both) - w - l))
    print("后者更好的局:", [g for g in both if b[g]["score"] > a[g]["score"]] or "无")
    print("前者更好的局:", [g for g in both if b[g]["score"] < a[g]["score"]] or "无")
    print("\n%-6s %-18s %-18s" % ("局", la, lb))
    for g in both:
        print("%-6s 分%-5.2f 步%-4d 关%-3d 分%-5.2f 步%-4d 关%-3d"
              % (g, a[g]["score"], a[g]["actions"], a[g]["levels"],
                 b[g]["score"], b[g]["actions"], b[g]["levels"]))


if __name__ == "__main__":
    main()
