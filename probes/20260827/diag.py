"""第六发(LB-9复刻, 公榜1.92)逐局诊断: 步数账/轮次账/时间账/animation收益。

数据源: probes/20260826/lb9_out/{arc3-duck-q38-lb9.log, transcripts/*.txt}
      + probes/20260827/baseline_actions.json (各关官方基准步数)
"""
import re, json, glob, statistics, collections, os
from datetime import datetime

os.chdir(os.path.expanduser("~/Desktop/project/arc-agi-3"))
LOG = "probes/20260826/lb9_out/arc3-duck-q38-lb9.log"
TDIR = "probes/20260826/lb9_out/transcripts"

hp = re.compile(r'^--- analysis_step=(\d+) \| action=(\d+) \| (\d+:\d+:\d+)')
lv = re.compile(r'Current state: step \d+, level (\d+)')

# --- 日志: 逐局成绩 + 每关实际/基准步数 ---
games = {}
for line in open(LOG):
    line = line.strip().lstrip(',')
    try: d = json.loads(line)
    except Exception: continue
    m = re.search(r'\[finished\] (\S+?)-(\S+) state=(\S+) level=(\d+)/(\d+) score=([\d.]+) actions=(\d+) tokens=(\d+) per-level=(\S+)', d.get('data',''))
    if m:
        games[m.group(1)] = dict(gid=f"{m.group(1)}-{m.group(2)}", state=m.group(3), lv=int(m.group(4)),
                                 tot=int(m.group(5)), sc=float(m.group(6)), act=int(m.group(7)),
                                 pl=[tuple(int(x) for x in p.split('/')) for p in m.group(9).split(',')])

# --- transcript: 轮次/时间/动作/animation ---
for f in sorted(glob.glob(f"{TDIR}/*_p0.txt")):
    g = os.path.basename(f)[:4]
    if g not in games: continue
    t = open(f, encoding='utf-8', errors='replace').read()
    segs = re.split(r'(?=^--- analysis_step=)', t, flags=re.M)[1:]
    if not segs:
        continue  # 空/损坏的 transcript, 跳过
    heads = [hp.match(s) for s in segs]
    times, levels = [], []
    for h, s in zip(heads, segs):
        times.append(datetime.strptime(h.group(3), '%H:%M:%S') if h else None)
        mm = lv.search(s); levels.append(int(mm.group(1)) if mm else None)
    gaps = []
    for i in range(len(times)-1):
        if times[i] and times[i+1]:
            gp = (times[i+1]-times[i]).total_seconds()
            gaps.append(gp if gp >= 0 else gp+86400)
    deltas = [int(heads[i+1].group(2))-int(heads[i].group(2))
              for i in range(len(heads)-1) if heads[i] and heads[i+1]]
    # 过关轮
    up, last = set(), None
    for i, l in enumerate(levels):
        if l is None: continue
        if last is not None and l > last: up.add(i)
        last = l
    anim = animwin = 0
    kinds = collections.Counter()
    for i, s in enumerate(segs):
        m = re.search(r'\[TOOL CALL[^\]]*\]\n(.*?)(?=\n\[TOOL RESULT|\Z)', s, re.S)
        code = m.group(1) if m else ''
        if re.search(r'(?<![a-z_])animation\(', code):
            anim += 1
            if any(j in up for j in range(i+1, min(i+5, len(segs)))): animwin += 1
        if i < len(deltas) and deltas[i] == 0:
            if 'action(' in code: kinds['调action没走成'] += 1
            elif re.search(r'(?<![a-z_])animation\(', code): kinds['animation回看'] += 1
            elif 'segmentation' in code: kinds['读segmentation'] += 1
            elif '.ascii' in code: kinds['读ascii'] += 1
            else: kinds['纯计算/其他'] += 1
    games[g].update(turns=len(segs), gap=statistics.median(gaps) if gaps else 0,
                    deltas=deltas, anim=anim, animwin=animwin, kinds=kinds,
                    reas=sum(int(x) for x in re.findall(r'^reasoning_chars: (\d+)', t, re.M)),
                    cont=sum(int(x) for x in re.findall(r'^content_chars: (\d+)', t, re.M)))

bl = json.load(open("probes/20260827/baseline_actions.json"))
have = [g for g in games if 'turns' in games[g]]

print(f"=== 第六发逐局诊断 (公榜 1.92, 25局全部 gave_up=时间到)   transcript 覆盖 {len(have)}/25 局\n")
print(f"{'局':5}{'分':>6}{'关':>3}{'步':>5}{'轮':>4}{'秒/轮':>6}{'零动作轮':>8}{'动作/轮':>7}{'思考字符/轮':>10}{'anim':>5}{'通关需要':>8}{'走完':>7}")
rows = []
for g in sorted(have, key=lambda x: -games[x]['sc']):
    d = games[g]; z = sum(1 for x in d['deltas'] if x == 0)
    need = sum(bl.get(g, [])) or 0
    rows.append((g, d, z, need))
    print(f"{g:5}{d['sc']:>6.2f}{d['lv']:>3}{d['act']:>5}{d['turns']:>4}{d['gap']:>6.0f}"
          f"{100*z/max(1,len(d['deltas'])):>7.0f}%{d['act']/d['turns']:>7.2f}{d['reas']/d['turns']:>10.0f}"
          f"{d['anim']:>5}{need:>8}{100*d['act']/need if need else 0:>6.1f}%")

alld = [x for _, d, _, _ in rows for x in d['deltas']]
allk = collections.Counter()
for _, d, _, _ in rows: allk.update(d['kinds'])
zt = sum(z for _, _, z, _ in rows); tt = sum(len(d['deltas']) for _, d, _, _ in rows)
print(f"\n=== 合计 ({len(rows)} 局)")
print(f"  每轮耗时中位 {statistics.median([d['gap'] for _, d, _, _ in rows]):.0f} 秒 → 一局 7920 秒只够 {7920/statistics.median([d['gap'] for _, d, _, _ in rows]):.0f} 轮")
print(f"  每轮思考 {sum(d['reas'] for _, d, _, _ in rows)/sum(d['turns'] for _, d, _, _ in rows):.0f} 字符 | 回复 {sum(d['cont'] for _, d, _, _ in rows)/sum(d['turns'] for _, d, _, _ in rows):.0f} 字符")
print(f"  零动作轮 {zt}/{tt} = {100*zt/tt:.1f}%")
for k, v in allk.most_common(): print(f"      {k:16} {v:4}  {100*v/zt:5.1f}%")
print(f"  每轮动作数: 中位 {statistics.median(alld):.0f}, 平均 {sum(alld)/len(alld):.2f}, 最大 {max(alld)}")
print(f"  animation() 调用 {sum(d['anim'] for _, d, _, _ in rows)} 次, 之后4轮内过关 {sum(d['animwin'] for _, d, _, _ in rows)} 次")
tn = sum(need for _, _, _, need in rows); ta = sum(d['act'] for _, d, _, _ in rows)
print(f"  步数总账: 实走 {ta} / 通关需要 {tn} = {100*ta/tn:.1f}%")
stuck = [(g, d['pl'][d['lv']][0], d['pl'][d['lv']][1]) for g, d, _, _ in rows if d['lv'] < len(d['pl']) and d['pl'][d['lv']][0] > 0]
if stuck:
    r = [a/b for _, a, b in stuck]
    print(f"  卡住那关: 实际/基准 中位 {statistics.median(r):.2f}, 连基准都没走到的 {sum(1 for x in r if x<1)}/{len(r)} 局")
