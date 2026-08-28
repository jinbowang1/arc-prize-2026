# -*- coding: utf-8 -*-
"""第一层 A/B 结果分析。按 notes/experiment-protocol-20260827.md 的判据:
机制指标为主(零动作轮/每轮动作数/每轮思考字符), 分数只作参考。

用法: ab_report.py <对照run目录> <实验run目录> [实验名]
"""
import sys, os, re, glob, math, statistics, collections

ROOT = os.path.expanduser("~/Desktop/project/arc-agi-3/probes/20260825/duck_guide_ab")
hp = re.compile(r'^--- analysis_step=(\d+) \| action=(\d+) \| (\d+:\d+:\d+)')
lvre = re.compile(r'Current state: step \d+, level (\d+)')


def collect(run):
    d = run if os.path.isdir(run) else os.path.join(ROOT, run)
    out = {}
    for f in sorted(glob.glob(f"{d}/transcripts/*.txt")):
        key = os.path.basename(f).replace("_p", "|").replace(".txt", "")
        t = open(f, encoding="utf-8", errors="replace").read()
        segs = re.split(r'(?=^--- analysis_step=)', t, flags=re.M)[1:]
        if not segs:
            continue
        heads = [hp.match(s) for s in segs]
        deltas = [int(heads[i+1].group(2)) - int(heads[i].group(2))
                  for i in range(len(heads)-1) if heads[i] and heads[i+1]]
        levels = [int(m.group(1)) if (m := lvre.search(s)) else None for s in segs]
        known = [l for l in levels if l is not None]
        out[key] = dict(
            turns=len(segs), deltas=deltas,
            acts=max([int(h.group(2)) for h in heads if h] + [0]),
            zero=sum(1 for x in deltas if x == 0),
            reas=[int(x) for x in re.findall(r'^reasoning_chars: (\d+)', t, re.M)],
            code=sum(len(m.group(1)) for m in re.finditer(r'\[TOOL CALL[^\]]*\]\n(.*?)(?=\n\[TOOL RESULT|\Z)', t, re.S)),
            think=sum(len(m.group(1)) for m in re.finditer(r'\[THINKING\]\n(.*?)(?=\n\[[A-Z])', t, re.S)),
            nums=sum(len(re.findall(r'\b\d+\b', m.group(1))) for m in re.finditer(r'\[THINKING\]\n(.*?)(?=\n\[[A-Z])', t, re.S)),
            maxlevel=max(known) if known else 0,
        )
    return out


def two_prop_z(k1, n1, k2, n2):
    """两比例 z 检验, 返回 (差值百分点, z, p)"""
    if not n1 or not n2:
        return (0.0, 0.0, 1.0)
    p1, p2 = k1/n1, k2/n2
    p = (k1+k2)/(n1+n2)
    se = math.sqrt(p*(1-p)*(1/n1 + 1/n2))
    if se == 0:
        return ((p2-p1)*100, 0.0, 1.0)
    z = (p2-p1)/se
    p_val = math.erfc(abs(z)/math.sqrt(2))
    return ((p2-p1)*100, z, p_val)


def main():
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    ctrl, exp = collect(sys.argv[1]), collect(sys.argv[2])
    name = sys.argv[3] if len(sys.argv) > 3 else "实验"
    shared = sorted(set(ctrl) & set(exp))
    print(f"配对到 {len(shared)} 个 局-遍 (对照 {len(ctrl)} / 实验 {len(exp)})")
    if not shared:
        print("⚠️ 没有可配对的局-遍, 两臂的局或遍数不一致"); sys.exit(1)

    def agg(d, field):
        return sum(d[k][field] for k in shared)
    cz, ct = agg(ctrl, 'zero'), sum(len(ctrl[k]['deltas']) for k in shared)
    ez, et = agg(exp, 'zero'), sum(len(exp[k]['deltas']) for k in shared)
    diff, z, p = two_prop_z(cz, ct, ez, et)

    print()
    print(f"{'指标':16}{'对照':>12}{'实验':>12}{'变化':>12}")
    print(f"{'零动作轮':16}{f'{100*cz/max(1,ct):.1f}%':>12}{f'{100*ez/max(1,et):.1f}%':>12}{f'{diff:+.1f}pp':>12}   p={p:.3f}{'  ✅显著' if p<0.05 else '  ✗不显著'}")
    ca, cturn = agg(ctrl, 'acts'), agg(ctrl, 'turns')
    ea, eturn = agg(exp, 'acts'), agg(exp, 'turns')
    print(f"{'每轮动作数':16}{ca/max(1,cturn):>12.2f}{ea/max(1,eturn):>12.2f}{f'{ea/max(1,eturn)-ca/max(1,cturn):+.2f}':>12}")
    cr = [x for k in shared for x in ctrl[k]['reas']]
    er = [x for k in shared for x in exp[k]['reas']]
    if cr and er:
        print(f"{'每轮思考字符':16}{statistics.median(cr):>12.0f}{statistics.median(er):>12.0f}{f'{statistics.median(er)-statistics.median(cr):+.0f}':>12}   (中位)")
    print(f"{'总轮数':16}{cturn:>12}{eturn:>12}")
    print(f"{'总步数':16}{ca:>12}{ea:>12}")
    cc, ct_ = agg(ctrl, 'code'), agg(ctrl, 'think')
    ec, et_ = agg(exp, 'code'), agg(exp, 'think')
    if ct_ and et_:
        print(f"{'代码/思考占比':16}{f'{100*cc/ct_:.1f}%':>12}{f'{100*ec/et_:.1f}%':>12}{f'{100*ec/et_-100*cc/ct_:+.1f}pp':>12}   ← 分工改动的直接指标")
        cn, en = agg(ctrl, 'nums'), agg(exp, 'nums')
        print(f"{'思考里数字密度':16}{f'{1000*cn/ct_:.1f}':>12}{f'{1000*en/et_:.1f}':>12}{f'{1000*en/et_-1000*cn/ct_:+.1f}':>12}   (每千字符)")
    cl, el = agg(ctrl, 'maxlevel'), agg(exp, 'maxlevel')
    print(f"{'总过关数':16}{cl:>12}{el:>12}{f'{el-cl:+d}':>12}   (参考)")

    print()
    print(f"样本量核对: 每臂需 ≥400 轮才能检出 10 个百分点 —— "
          f"对照 {cturn} 轮 {'✅' if cturn>=400 else '⚠️不足'}, 实验 {eturn} 轮 {'✅' if eturn>=400 else '⚠️不足'}")
    print()
    print(f"逐局配对 (零动作轮%):")
    wins = losses = 0
    for k in shared:
        c_r = 100*ctrl[k]['zero']/max(1, len(ctrl[k]['deltas']))
        e_r = 100*exp[k]['zero']/max(1, len(exp[k]['deltas']))
        mark = '↓' if e_r < c_r else ('↑' if e_r > c_r else '=')
        if e_r < c_r: wins += 1
        elif e_r > c_r: losses += 1
        print(f"  {k:24} {c_r:5.1f}% → {e_r:5.1f}%  {mark}")
    print(f"  配对结果: {name}更低 {wins} 个, 更高 {losses} 个")


if __name__ == "__main__":
    main()
