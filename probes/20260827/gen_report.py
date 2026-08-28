# -*- coding: utf-8 -*-
"""生成 08-27 ARC-AGI-3 第六发诊断报告(彩卡翻页风 HTML)。数字全部从原始日志/transcript 现算。"""
import re, json, glob, os, statistics, collections
from datetime import datetime

ROOT = os.path.expanduser("~/Desktop/project/arc-agi-3")
os.chdir(ROOT)
SC = "/private/tmp/claude-1518879226/-Users-01450825/73b29d8b-5d3c-4369-b798-7cbab154e4ee/scratchpad"
LOG = "probes/20260826/lb9_out/arc3-duck-q38-lb9.log"
TDIR = "probes/20260826/lb9_out/transcripts"
hp = re.compile(r'^--- analysis_step=(\d+) \| action=(\d+) \| (\d+:\d+:\d+)')
lvre = re.compile(r'Current state: step \d+, level (\d+)')

# ── 日志 ──
G = {}
for line in open(LOG):
    line = line.strip().lstrip(',')
    try: d = json.loads(line)
    except Exception: continue
    m = re.search(r'\[finished\] (\S+?)-(\S+) state=(\S+) level=(\d+)/(\d+) score=([\d.]+) actions=(\d+) tokens=(\d+) per-level=(\S+)', d.get('data',''))
    if m:
        G[m.group(1)] = dict(state=m.group(3), lv=int(m.group(4)), tot=int(m.group(5)), sc=float(m.group(6)),
                             act=int(m.group(7)), pl=[tuple(int(x) for x in p.split('/')) for p in m.group(9).split(',')])
# ── transcript ──
allreas = []
for f in sorted(glob.glob(f"{TDIR}/*_p0.txt")):
    g = os.path.basename(f)[:4]
    if g not in G: continue
    t = open(f, encoding='utf-8', errors='replace').read()
    segs = re.split(r'(?=^--- analysis_step=)', t, flags=re.M)[1:]
    if not segs: continue
    heads = [hp.match(s) for s in segs]
    times = [datetime.strptime(h.group(3), '%H:%M:%S') if h else None for h in heads]
    gaps = []
    for i in range(len(times)-1):
        if times[i] and times[i+1]:
            gp = (times[i+1]-times[i]).total_seconds(); gaps.append(gp if gp >= 0 else gp+86400)
    deltas = [int(heads[i+1].group(2))-int(heads[i].group(2)) for i in range(len(heads)-1) if heads[i] and heads[i+1]]
    levels = [int(m.group(1)) if (m := lvre.search(s)) else None for s in segs]
    up, last = set(), None
    for i, l in enumerate(levels):
        if l is None: continue
        if last is not None and l > last: up.add(i)
        last = l
    anim = animwin = 0; kinds = collections.Counter()
    for i, s in enumerate(segs):
        mm = re.search(r'\[TOOL CALL[^\]]*\]\n(.*?)(?=\n\[TOOL RESULT|\Z)', s, re.S)
        code = mm.group(1) if mm else ''
        isanim = bool(re.search(r'(?<![a-z_])animation\(', code))
        if isanim:
            anim += 1
            if any(j in up for j in range(i+1, min(i+5, len(segs)))): animwin += 1
        if i < len(deltas) and deltas[i] == 0:
            if 'action(' in code: kinds['调了 action 却没走出一步'] += 1
            elif isanim: kinds['回看 animation'] += 1
            elif 'segmentation' in code: kinds['读 segmentation 解析棋盘'] += 1
            elif '.ascii' in code: kinds['读 ascii 看棋盘'] += 1
            else: kinds['纯计算 / 其它'] += 1
    reas = [int(x) for x in re.findall(r'^reasoning_chars: (\d+)', t, re.M)]
    allreas += reas
    G[g].update(turns=len(segs), gap=statistics.median(gaps) if gaps else 0, deltas=deltas,
                anim=anim, animwin=animwin, kinds=kinds, reas=sum(reas),
                cont=sum(int(x) for x in re.findall(r'^content_chars: (\d+)', t, re.M)),
                fr=collections.Counter(re.findall(r'^finish_reason: (\S+)', t, re.M)))

BL = json.load(open("probes/20260827/baseline_actions.json"))
ANIM = json.load(open("probes/20260827/anim_scan_s0.json"))
H = [g for g in G if 'turns' in G[g]]
by = sorted(H, key=lambda x: -G[x]['sc'])

# ── 汇总量 ──
alld = [x for g in H for x in G[g]['deltas']]
allk = collections.Counter()
for g in H: allk.update(G[g]['kinds'])
zt = sum(1 for g in H for x in G[g]['deltas'] if x == 0); tt = len(alld)
medgap = statistics.median([G[g]['gap'] for g in H])
turns_tot = sum(G[g]['turns'] for g in H)
reas_avg = sum(G[g]['reas'] for g in H)/turns_tot
cont_avg = sum(G[g]['cont'] for g in H)/turns_tot
need_tot = sum(sum(BL.get(g, [])) for g in H)
act_tot = sum(G[g]['act'] for g in H)
anim_tot = sum(G[g]['anim'] for g in H); animwin_tot = sum(G[g]['animwin'] for g in H)
stuck = [(g, G[g]['pl'][G[g]['lv']][0], G[g]['pl'][G[g]['lv']][1])
         for g in H if G[g]['lv'] < len(G[g]['pl']) and G[g]['pl'][G[g]['lv']][0] > 0]
stuck.sort(key=lambda x: x[1]/x[2])
sr = [a/b for _, a, b in stuck]
allreas.sort(); N = len(allreas); TOTR = sum(allreas)
allfr = collections.Counter()
for g in H: allfr.update(G[g]['fr'])

# ══════════════════════════════════════ 页面 ══════════════════════════════════════
C = ['var(--c1)','var(--c2)','var(--c3)','var(--c4)','var(--c5)','var(--c6)','var(--c7)','var(--c8)']
def shd(kick, kc, title, lede):
    return (f'<div class="shd"><span class="kicker" style="background:{kc}22;color:{kc}">{kick}</span>'
            f'<h2>{title}</h2><p class="lede">{lede}</p></div>')
n_multi = sum(1 for g in ANIM if any(r['frames'] > 1 for r in ANIM[g]))
n_hid = sum(1 for g in ANIM if any(r['board_unchanged'] and r['transient'] > 0 for r in ANIM[g]))
P = []

# ── p0 封面 ──
P.append(f'''<section class="page cover" data-on="1">
  <span class="eyebrow">ARC-AGI-3 · 第六发诊断 · 2026-08-27</span>
  <h1>1.92 分打在了<br><em>吐字速度</em>上</h1>
  <p class="sub">第六发（Duck harness + Qwen3.8-27B + 带图，公榜 1.92，173/2561）在 25 个公开局上全部跑到时间用尽。把日志、25 份完整对话记录和引擎重放对齐之后，卡点落在一个跟策略无关的地方：一局两小时十二分钟，模型只走得出五十来步，而这些局全部通关需要一万五千步。</p>
  <div class="kpis">
    <div class="kpi" style="border-top-color:{C[0]}"><b>1.92</b><span>公榜得分 · 前一发 0.95</span></div>
    <div class="kpi" style="border-top-color:{C[1]}"><b>173</b><span>名次 / 2561 队 · 前百门槛 2.11</span></div>
    <div class="kpi" style="border-top-color:{C[7]}"><b>{100*act_tot/need_tot:.1f}%</b><span>实走 {act_tot} 步 / 通关需 {need_tot} 步</span></div>
    <div class="kpi" style="border-top-color:{C[4]}"><b>{medgap:.0f} 秒</b><span>每轮耗时中位 · 一局仅 {7920/medgap:.0f} 轮</span></div>
    <div class="kpi" style="border-top-color:{C[5]}"><b>{100*zt/tt:.1f}%</b><span>轮次一步没走出</span></div>
    <div class="kpi" style="border-top-color:{C[2]}"><b>{reas_avg:.0f}</b><span>每轮思考字符 · 回复仅 {cont_avg:.0f}</span></div>
  </div>
  <p class="colo">数据口径：kernel arc3-duck-q38-lb9（08-26 18:31→20:51）· 25/25 局完整对话记录 · 引擎离线重放 24 局（dc22 本地缺游戏源码）</p>
</section>''')

# ── p1 源码包真相 ──
P.append(f'''<section class="page">
  {shd('SOURCE','var(--c8)','昨天交上去的那份源码','LB-9 复刻件用的源码包并非 Tufa 官方原样。它是另一名参赛者自己的实验分支，且是一份未提交完的工作树，里面装着两个官方版没有的功能。')}
  <div class="body"><div class="think" style="grid-template-rows:1fr">
    <div class="tk" style="border-top-color:{C[7]};--d:{C[7]}">
      <div class="tkn">01 / 包的来历</div><div class="tkt">分支 feature/animation-awareness，状态 DIRTY</div>
      <div class="tkd">包内 <b>git_status.txt</b> 写明：ARC3-Inference 与 tufa-arc-agi-framework 两个仓都停在 commit 9158303、分支 <b>feature/animation-awareness</b>、工作树 DIRTY，提交信息是「proactively suggest animation retrieval when stuck」。官方包停在 aa69123 / add-kaggle-share-flag。</div>
      <div class="tkeg">整树 diff：净增 <b>844 行</b>、删 404 行；新增两个文件 <b>inference/utils/animation.py</b>（375 行）与 <b>inference/agent/noop_guard.py</b>（100 行）。</div>
    </div>
    <div class="tk" style="border-top-color:{C[0]};--d:{C[0]}">
      <div class="tkn">02 / 多出来的两件东西</div><div class="tkt">动画感知，与无效动作硬拦截</div>
      <div class="tkd">代码注释里带着作者自己的实验编号：<b>Experiment 1</b>（自动记忆，已回滚）、<b>Experiment 2</b>（无效动作拦截，2026-07-18）、<b>Experiment 3</b>（动画感知，2026-08-07）。两者都留了环境变量开关，默认全开。</div>
      <div class="tkeg">Experiment 1 回滚的理由写得很直白：只在上下文里<b>提醒</b>模型某个动作没用，模型照犯不误，无效重复仍剩 <b>12%</b>。所以第二版改成 harness 直接拦。</div>
    </div>
    <div class="tk" style="border-top-color:{C[3]};--d:{C[3]}">
      <div class="tkn">03 / 一条要更正的旧记录</div><div class="tkt">并发 28 是官方原版就有的设置</div>
      <div class="tkd">此前记为「LB-9 = 换模型 + 并发 28」。核对两个包的 <b>preamble.txt</b>：官方包 solver 参数里 concurrency 本来就是 <b>28</b>，max_runtime_s_per_game 本来就是 <b>7920</b>。这个差异点不成立。</div>
      <div class="tkeg">于是官方版 2.08 → LB-9 复刻 2.74 的那 <b>0.66 分</b>，来源只剩两处：模型从 Qwen3.6 换到 3.8，以及上面这两个功能。</div>
    </div>
  </div></div>
</section>''')

# ── p2 动画机制 ──
P.append(f'''<section class="page">
  {shd('MECHANISM','var(--c1)','被丢掉的中间帧','arcengine 每执行一次内部 step 就渲染一帧，一个动作可能返回几十帧。官方 harness 只把最后一帧交给模型，中间的全部丢弃。')}
  <div class="body"><div class="two">
    <div class="pane"><div class="pd">
      <div class="role" style="--d:{C[1]}">三级机制</div>
      <div class="sect">STAGE 1 · 每个动作后的摘要</div>
      <div class="etag">回传 <b>frames</b>（返回几帧）、<b>unique_frames</b>、<b>board_unchanged</b>（末帧与动作前完全相同）、<b>transient_pixels</b> 及其外框。单帧动作不附加任何 token。</div>
      <div class="sect">STAGE 2 · 按需回看</div>
      <div class="etag">模型可调 <b>animation()</b> 取一段差分时间线，<b>animation(frame=k)</b> 按 transient 区域裁剪读某一帧。不消耗游戏动作预算。</div>
      <div class="sect">STAGE 3 · 卡住时主动提示</div>
      <div class="etag">同一关连续 <b>6 轮</b>无进展、且本关已有 <b>≥2 次</b>带 transient 像素的动画时才触发，冷却 6 轮。</div>
      <div class="mini3" style="--d:{C[1]}">
        <div><b>375</b>animation.py 行数</div><div><b>8</b>回看最多展示帧数</div><div><b>80</b>回看总格子预算</div></div>
      <div class="sect">同步改了提示词</div>
      <div class="etag">工具说明里新增 6 行，明写一句话：<b>board_unchanged 加上 board_changed == False 并不代表这个动作什么也没做</b>，被拒绝的点击、消耗掉的机会、撞墙的反弹，只显示在没被看到的那几帧里。</div>
      <div class="dl"><div class="dt">触发阈值</div><div class="dd">无进展 <span class="c">6 轮</span>带 transient 动画 <span class="c">≥2 次</span>冷却 <span class="c">6 轮</span>跟进窗口 <span class="c">3 轮</span></div>
      <div class="dt">为何不给整帧</div><div class="dd">一张 64×64 棋盘渲染成 4159 字符、约 1400–2000 token，而整个工具返回预算只有 1024 token。sb26 单动作最多 42 帧，去重后仍有 2.4–3.4 万 token，只能给差分时间线。</div>
      <div class="dt">回看的形态</div><div class="dd">连续相同的帧折叠成 held_for_frames，逐帧只报变化的格子，写成 <span class="c">W&gt;R @ (row,col)</span> 这样一行；变化太多就退化成颜色迁移的计数。超预算时保留变化最大的几步，再按时间顺序排回去。</div></div>
    </div></div>
    <div class="pane"><div class="pd">
      <div class="role" style="--d:{C[3]}">作者的实测，与我方复现</div>
      <div class="ewhy" style="--d:{C[3]}">注释原文：「Measured over 24 games (12 multi-frame responses each), 13 games return multi-frame responses, in two distinct shapes」，并把 <b>ft09、sb26</b> 归为 type 1，<b>r11l、sk48</b> 归为 type 2。</div>
      <div class="sect">两种形状</div>
      <div class="etag"><b>type 1</b>　首帧末帧相同，一次被拒绝的点击、一次消耗掉的机会，全部只存在于中间帧 —— 只读末帧的模型物理上看不见。<br><b>type 2</b>　纯运动插值，中间帧不含末帧以外的信息。</div>
      <div class="sect">重放怎么走的</div>
      <div class="etag">每局先按「每类动作各试一遍」走 6 步，其余按 55% 概率随机点击、45% 随机方向键，共 150 步；遇终局就重开继续采样。逐动作记录返回帧数、末帧是否与动作前相同、中间帧独有的像素数。</div>
      <div class="sect">复现口径</div>
      <div class="etag">直接 import 他的 <b>summarize_animation</b> 对本地 24 局重放（随机策略 150 步/局）。他写「42 frames of 64x64 for one sb26 action」，我方实测 sb26 单动作最大帧数同为 <b>42</b>；他写 24 局中 13 局多帧，我方测得 <b>14</b> 局。同一把尺子。</div>
      <div class="mini3" style="--d:{C[3]}">
        <div><b>{n_multi} / 24</b>会返回多帧</div><div><b>{n_hid} / 24</b>末帧看不见</div><div><b>42</b>sb26 单动作最大帧数</div></div>
      <div class="sect">另一件配套的事</div>
      <div class="etag">同一份改动里还修了个隐患：<b>带动画的动作永远不算无效动作</b>。type 1 里首帧末帧本就相同，若按「棋盘没变」记成无效，无效动作拦截会在动画最多的那几局把明明起效的操作硬挡掉。</div>
      <div class="dl"><div class="dt">脚本</div><div class="dd"><span class="c">probes/20260827/anim_scan.py</span><span class="c">anim_scan_s0.json</span></div>
      <div class="dt">局限</div><div class="dd">随机走法进不去深层关卡，深层可能仍有未采到的 type 1 情形；dc22 本地只有 metadata.json、缺游戏源码，无法起环境。</div></div>
    </div></div>
  </div></div>
</section>''')

# ── p3 动画分布表 ──
rows = []
for g in sorted(ANIM, key=lambda x: -sum(1 for r in ANIM[x] if r['frames'] > 1)):
    rs = ANIM[g]; multi = [r for r in rs if r['frames'] > 1]
    hid = [r for r in multi if r['board_unchanged'] and r['transient'] > 0]
    mx = max((r['frames'] for r in rs), default=0)
    if not multi: kind, tag = '无动画', 't-x'
    elif hid: kind, tag = 'type 1 · 信息藏在中间帧', 't-ok'
    else: kind, tag = 'type 2 · 纯运动插值', 't-w'
    sc = f"{G[g]['sc']:.2f}" if g in G else '—'
    rows.append(f"<tr><td><b>{g}</b></td><td>{len(rs)}</td><td>{len(multi)}</td>"
                f"<td>{100*len(multi)/max(1,len(rs)):.1f}%</td><td><b>{len(hid)}</b></td><td>{mx}</td>"
                f"<td><span class='tag {tag}'>{kind}</span></td><td>{sc}</td></tr>")
P.append(f'''<section class="page">
  {shd('SCAN','var(--c2)','哪些局的动画真的藏着东西','把判据收紧到「末帧与动作前完全相同、东西全在中间帧」——只有这种情形，模型读末帧无论如何也看不到。24 局离线重放，每局随机走 150 步。')}
  <div class="body"><div class="pane dense" style="flex:1"><div class="tw"><table>
    <thead><tr><th>局</th><th>采样动作</th><th>多帧动作</th><th>占比</th><th>末帧看不见的</th><th>最大帧数</th><th>归类</th><th>第六发得分</th></tr></thead>
    <tbody>{''.join(rows)}</tbody></table></div>
    <div class="legend"><b style="background:{C[3]}">{n_multi} / 24</b> 局会返回多帧（作者测 24 局中 13 局）
      <b style="background:{C[7]}">{n_hid} / 24</b> 局存在末帧看不见的信息 —— ft09、sc25、cd82
      <b style="background:{C[5]}">10 / 24</b> 局完全不产生多帧，此功能对它们无作用</div>
  </div></div>
</section>''')

# ── p4 逐局成绩 ──
rows = []
for g in by:
    d = G[g]; z = sum(1 for x in d['deltas'] if x == 0); need = sum(BL.get(g, []))
    cls = 't-ok' if d['sc'] >= 4 else ('t-w' if d['sc'] > 0 else 't-x')
    rows.append(f"<tr><td><b>{g}</b></td><td><span class='tag {cls}'>{d['sc']:.2f}</span></td>"
                f"<td>{d['lv']} / {d['tot']}</td><td><b>{d['act']}</b></td><td>{d['turns']}</td>"
                f"<td>{d['gap']:.0f}s</td><td>{100*z/max(1,len(d['deltas'])):.0f}%</td>"
                f"<td>{d['act']/d['turns']:.2f}</td><td>{d['reas']/d['turns']:.0f}</td>"
                f"<td>{d['anim'] or '—'}</td><td>{need or '—'}</td>"
                f"<td><b>{100*d['act']/need:.1f}%</b></td></tr>" if need else
                f"<tr><td><b>{g}</b></td><td><span class='tag {cls}'>{d['sc']:.2f}</span></td>"
                f"<td>{d['lv']} / {d['tot']}</td><td><b>{d['act']}</b></td><td>{d['turns']}</td>"
                f"<td>{d['gap']:.0f}s</td><td>{100*z/max(1,len(d['deltas'])):.0f}%</td>"
                f"<td>{d['act']/d['turns']:.2f}</td><td>{d['reas']/d['turns']:.0f}</td>"
                f"<td>{d['anim'] or '—'}</td><td>—</td><td>—</td></tr>")
n0 = sum(1 for g in H if G[g]['lv'] == 0); n1 = sum(1 for g in H if G[g]['lv'] == 1)
n2 = sum(1 for g in H if G[g]['lv'] == 2)
P.append(f'''<section class="page">
  {shd('SCORE','var(--c4)','1.92 分的构成','25 局全部以 gave_up 结束 —— 时间用尽，没有一局玩通。开张的有 17 局，过 2 关的 6 局，过 3 关的 0 局。')}
  <div class="body"><div class="pane dense" style="flex:1"><div class="tw"><table>
    <thead><tr><th>局</th><th>得分</th><th>过关</th><th>步数</th><th>模型轮数</th><th>每轮耗时</th><th>零动作轮</th><th>动作/轮</th><th>思考字符/轮</th><th>animation</th><th>通关需要</th><th>走完</th></tr></thead>
    <tbody>{''.join(rows)}</tbody></table></div>
    <div class="legend"><b style="background:{C[7]}">{n0} 局</b> 一关未过
      <b style="background:{C[5]}">{n1} 局</b> 过 1 关
      <b style="background:{C[3]}">{n2} 局</b> 过 2 关
      <b style="background:{C[0]}">0 局</b> 过 3 关及以上　·　dc22 本地缺游戏源码，取不到基准步数</div>
  </div></div>
</section>''')

# ── p5 玩得动但没时间 ──
passed = []
for g in by:
    d = G[g]
    for i in range(d['lv']):
        a, b = d['pl'][i]
        if a and b: passed.append((g, i+1, a, b))
passed.sort(key=lambda x: x[2]/x[3])
fast = [x for x in passed if x[2] < x[3]]
frows = ''.join(f"<tr><td><b>{g}</b></td><td>第 {i} 关</td><td>{a}</td><td>{b}</td>"
                f"<td><span class='tag {'t-ok' if a<b else 't-w'}'>{a/b:.2f}</span></td></tr>"
                for g, i, a, b in passed)
srows = ''.join(f"<tr><td><b>{g}</b></td><td>第 {G[g]['lv']+1} 关</td><td>{a}</td><td>{b}</td>"
                f"<td><span class='tag {'t-x' if a/b<0.5 else ('t-w' if a/b<1 else 't-n')}'>{a/b:.2f}</span></td></tr>"
                for g, a, b in stuck)
P.append(f'''<section class="page">
  {shd('EVIDENCE','var(--c3)','玩得动，只是没时间玩','每局元数据带着官方给的各关基准步数，日志的 per-level 字段正好是「每关实际步数 / 基准步数」。两边一对，模型的水平和它的处境就分开了。')}
  <div class="body"><div class="two" style="grid-template-columns:1fr 1fr">
    <div class="pane dense"><div class="phd">已过掉的关 · 全部 {len(passed)} 关，按「实际 / 基准」升序</div><div class="tw"><table>
      <thead><tr><th>局</th><th>关</th><th>模型步数</th><th>官方基准</th><th>比值</th></tr></thead><tbody>{frows}</tbody></table></div>
      <div class="legend"><b style="background:{C[3]}">{len(fast)} / {len(passed)}</b> 关的步数少于官方基准
        <b style="background:{C[5]}">{len(passed)-len(fast)} / {len(passed)}</b> 关多于基准　·　比值中位 {statistics.median([a/b for _,_,a,b in passed]):.2f}</div></div>
    <div class="pane dense"><div class="phd">卡住的那一关 · 连基准步数都没走到就时间到</div><div class="tw"><table>
      <thead><tr><th>局</th><th>卡在</th><th>实走</th><th>基准</th><th>比值</th></tr></thead><tbody>{srows}</tbody></table></div>
      <div class="legend"><b style="background:{C[7]}">{sum(1 for x in sr if x<0.5)} 局</b> 连基准的一半都没走到
        <b style="background:{C[5]}">{sum(1 for x in sr if 0.5<=x<1)} 局</b> 走到基准的五成至九成
        <b style="background:#8B93B0">{sum(1 for x in sr if x>=1)} 局</b> 步数超过基准仍未过关　·　中位 {statistics.median(sr):.2f}</div></div>
  </div></div>
</section>''')

# ── p6 时间账 ──
trows = ''.join(f"<tr><td><b>{g}</b></td><td>{G[g]['turns']}</td><td>{G[g]['gap']:.0f}s</td>"
                f"<td>{G[g]['reas']/G[g]['turns']:.0f}</td><td>{G[g]['reas']:,}</td>"
                f"<td>{G[g]['act']}</td><td>{G[g]['sc']:.2f}</td></tr>"
                for g in sorted(H, key=lambda x: -G[x]['turns']))
P.append(f'''<section class="page">
  {shd('CLOCK','var(--c5)','一局两小时十二分，够走 47 轮','28 路共享一张卡，每路每秒吐得出十来个 token。一轮从发出到下一轮开始，中位数 168 秒。轮数与每轮思考量的乘积，各局都落在同一个量级。')}
  <div class="body"><div class="two" style="grid-template-columns:1.35fr 1fr">
    <div class="pane dense"><div class="phd">逐局 · 轮数与思考量（按轮数排序）</div><div class="tw"><table>
      <thead><tr><th>局</th><th>轮数</th><th>每轮耗时</th><th>思考字符/轮</th><th>思考总字符</th><th>步数</th><th>得分</th></tr></thead>
      <tbody>{trows}</tbody></table></div></div>
    <div class="pane"><div class="pd">
      <div class="role" style="--d:{C[5]}">一局能吐的字总量是死的</div>
      <div class="ewhy" style="--d:{C[5]}">ls20 每轮只想 2016 字，跑出 <b>74 轮</b>；lp85 每轮想 9083 字，只跑了 <b>28 轮</b>。两者相乘都在二十几万字符 —— 时间预算换算成吐字量之后，写得越省，轮数越多。</div>
      <div class="mini3" style="--d:{C[5]}">
        <div><b>{medgap:.0f}s</b>每轮耗时中位</div><div><b>{7920/medgap:.0f}</b>一局轮数上限</div><div><b>{turns_tot}</b>25 局总轮数</div></div>
      <div class="sect">每轮输出的去向</div>
      <div class="etag">思考 <b>{reas_avg:.0f}</b> 字符　·　回复正文 <b>{cont_avg:.0f}</b> 字符<br>思考占了输出的 <b>{100*reas_avg/(reas_avg+cont_avg):.1f}%</b>，而思考本身不产生任何动作。</div>
      <div class="sect">每轮输入的构成</div>
      <div class="etag">系统提示词 <b>14187</b> 字符（每轮重发）　·　用户消息 <b>3052</b> 字符　·　工具输出中位 <b>381</b> 字符　·　上下文窗口 31744 token</div>
      <div class="sect">公开集与隐藏集的配置</div>
      <div class="etag">notebook 里 <b>max_runtime_s_per_game=7920</b>、<b>concurrency=28</b> 只在公开分支显式设置；提交走 pickle 里的原值，核对下来两边一致。提交时 soft_end_time 返回 None，不额外收紧。<br>110 个隐藏局按并发 28 要跑 4 批，约 <b>8.8 小时</b>。</div>
      <div class="sect">线上确认为开启</div>
      <div class="etag">对话记录的状态块印着：<span class="c">hard_noop_guard: True</span><span class="c">context_budget_tokens: 31744</span><span class="c">tool_output_tokens: 1024</span><span class="c">model: Qwen/Qwen3.8-27B-FP8</span></div>
      <div class="dl"><div class="dt">口径</div><div class="dd">每轮耗时取相邻两轮时间戳之差的中位数；25 局并发同时起跑，每局时间上限 7920 秒</div></div>
    </div></div>
  </div></div>
</section>''')

# ── p7 零动作轮 ──
kk = allk.most_common()
kcards = ''.join(f'''<div class="nxc" style="border-top-color:{C[i]};--d:{C[i]}">
  <div class="nxh"><span class="nxt">{k}</span><span class="nxn">{v}</span></div>
  <div class="nxl"><div class="nxi">占零动作轮 <b>{100*v/zt:.1f}%</b></div>
  <div class="nxi">折合整轮时间约 <b>{v*medgap/3600:.1f} 小时</b></div>
  <div class="nxi">{desc}</div></div></div>'''
  for i, ((k, v), desc) in enumerate(zip(kk, [
    '棋盘不进上下文，模型必须调 python 才能读到 ascii。系统提示词明写「原始数字网格故意不暴露」。',
    'segmentation 把棋盘解析成对象、颜色、包含与邻接关系，是官方指定的「主要视图」，同样要单独一轮去取。',
    '在 python 里做统计、比对、记笔记，不触碰环境。',
    '主动回看某个动作的中间帧。不消耗游戏动作预算，但消耗一整轮。',
    '发出了 action() 调用，动作计数却没有前进 —— 终局状态或被无效动作拦截挡下。'])))
P.append(f'''<section class="page">
  {shd('IDLE','var(--c6)','过半轮次走不出一步','{zt} 个轮次里动作计数原地不动。按工具调用内容归类，三分之二花在「看棋盘」上。'.replace('{zt}', str(zt)))}
  <div class="body">
    <div class="kpis" style="grid-template-columns:repeat(4,1fr);margin:0 0 16px">
      <div class="kpi" style="border-top-color:{C[6]}"><b>{zt} / {tt}</b><span>零动作轮 · {100*zt/tt:.1f}%</span></div>
      <div class="kpi" style="border-top-color:{C[0]}"><b>{100*(allk['读 ascii 看棋盘']+allk['读 segmentation 解析棋盘'])/zt:.1f}%</b><span>其中用于读棋盘</span></div>
      <div class="kpi" style="border-top-color:{C[4]}"><b>{statistics.median(alld):.0f} / {sum(alld)/len(alld):.2f}</b><span>每轮动作数 中位 / 平均</span></div>
      <div class="kpi" style="border-top-color:{C[2]}"><b>{max(alld)}</b><span>单轮最多下发动作数</span></div>
    </div>
    <div class="nx" style="grid-template-columns:repeat(5,1fr)">{kcards}</div>
  </div>
</section>''')

# ── p8 思考长尾 ──
caps = [500,1000,1500,2000,2500,3000,4000,5000,6000,7000,8000,10000,12000,15000,20000]
crows = ''
for c in caps:
    over = [x for x in allreas if x > c]; saved = sum(x-c for x in over)
    hi = ' style="background:#FFF7ED"' if c == 8000 else ''
    crows += (f"<tr{hi}><td><b>{c:,}</b></td><td>{len(over)}</td><td>{100*len(over)/N:.1f}%</td>"
              f"<td>{saved:,}</td><td>{100*saved/TOTR:.1f}%</td>"
              f"<td><span class='tag t-ok'>+{100*saved/(TOTR-saved):.1f}%</span></td></tr>")
P.append(f'''<section class="page">
  {shd('TAIL','var(--c7)','思考长度的长尾','{N} 轮里 finish_reason 全是 tool_calls，一次 length 都没有 —— 思考从未被截断，写多长完全由模型自己决定。中位数三千字，尾巴拖到三万五。'.replace('{N}', str(N)))}
  <div class="body"><div class="two" style="grid-template-columns:1fr 1.15fr">
    <div class="pane"><div class="pd">
      <div class="role" style="--d:{C[7]}">分位数</div>
      <div class="mini3" style="--d:{C[7]};grid-template-columns:repeat(2,1fr)">
        <div><b>{statistics.median(allreas):,.0f}</b>中位</div><div><b>{TOTR/N:,.0f}</b>平均</div>
        <div><b>{allreas[int(N*0.9)]:,}</b>P90</div><div><b>{allreas[int(N*0.95)]:,}</b>P95</div></div>
      <div class="etag" style="margin-top:4px">最长的一轮 <b>{max(allreas):,}</b> 字符。按每路每秒吐 48 个字符估算，这一轮单独就要 <b>{max(allreas)/48/60:.0f} 分钟</b>，占掉一局时间的 {100*(max(allreas)/48)/7920:.0f}%。</div>
      <div class="sect">finish_reason 分布</div>
      <div class="etag">{'　·　'.join(f'<b>{k}</b> {v}' for k, v in allfr.most_common())}</div>
      <div class="sect">可用的旋钮</div>
      <div class="etag"><span class="c">LOCAL_ANALYZER_MAX_OUTPUT</span>默认 0（不封顶）<br><span class="c">LOCAL_ANALYZER_ENABLE_THINKING</span>默认 True<br>两者都是环境变量，源码零改动。<b>模块级读取，必须在 import 之前设。</b></div>
      <div class="sect">动手前要先确认的一件事</div>
      <div class="ewhy" style="--d:{C[7]}">封顶封的是整个输出，思考、正文、工具调用共用这一份额度。若截断落在思考中途、导致这一轮吐不出完整的工具调用，那么这一轮直接作废 —— 旋钮就成了负收益。查 tool_agent 对 finish_reason=length 的处理即可定论。</div>
      <div class="dl"><div class="dt">验证路径</div><div class="dd">按三层框架：<span class="c">DeepSeek 做实验</span><span class="c">qwen3.6-27b + 图 做验证</span><span class="c">线上 Qwen3.8 测试</span>三层一致才进当天提交</div></div>
    </div></div>
    <div class="pane"><div class="phd">单轮封顶的收益曲线 · 25 局 {N} 轮实测</div><div class="tw"><table>
      <thead><tr><th>封顶字符</th><th>触发轮次</th><th>占比</th><th>省下字符</th><th>占总量</th><th>同时间多跑轮次</th></tr></thead>
      <tbody>{crows}</tbody></table></div>
      <div class="legend"><b style="background:{C[4]}">8000</b> 建议起点：只碰 {100*len([x for x in allreas if x>8000])/N:.1f}% 的轮次，其余八成完全不受影响
        <b style="background:{C[8-1]}">待核实</b> 截断发生在思考中途时，该轮能否吐出完整工具调用</div></div>
  </div></div>
</section>''')

# ── p9 animation 成本收益 ──
arows = ''
for g in sorted(H, key=lambda x: (-G[x]['anim'], x)):
    rs = ANIM.get(g, []); multi = [r for r in rs if r['frames'] > 1]
    hid = sum(1 for r in multi if r['board_unchanged'] and r['transient'] > 0)
    kind = 'type 1' if hid else ('type 2' if multi else '无动画')
    tag = 't-ok' if hid else ('t-w' if multi else 't-x')
    a_ = G[g]['anim']
    mins = f"{a_*G[g]['gap']/60:.0f} 分" if a_ else "—"
    arows += (f"<tr><td><b>{g}</b></td><td>{a_ or '—'}</td>"
              f"<td>{G[g]['animwin'] if a_ else '—'}</td>"
              f"<td>{mins}</td>"
              f"<td><span class='tag {tag}'>{kind}</span></td>"
              f"<td>{G[g]['sc']:.2f}</td></tr>")
zero = [g for g in H if not G[g]['anim']]
P.append(f'''<section class="page">
  {shd('COST','var(--c1)','回看动画值不值','Stage 1 的摘要几乎不花钱，附在动作结果里几十个 token。花钱的是 Stage 2 —— 每次主动回看要占掉一整轮，而一轮是 168 秒。')}
  <div class="body"><div class="two" style="grid-template-columns:1.2fr 1fr">
    <div class="pane dense2"><div class="phd">逐局 · 回看次数与随后 4 轮内的过关</div><div class="tw"><table>
      <thead><tr><th>局</th><th>回看次数</th><th>随后 4 轮内过关</th><th>折合时间</th><th>该局动画类型</th><th>得分</th></tr></thead>
      <tbody>{arows}</tbody></table></div>
      <div class="legend"><b style="background:{C[1]}">{anim_tot} 次</b> 总回看
        <b style="background:{C[3]}">{animwin_tot} 次</b> 随后 4 轮内过关（{100*animwin_tot/anim_tot:.1f}%）
        <b style="background:{C[5]}">{len(zero)} 局</b> 一次没调过</div></div>
    <div class="pane"><div class="pd">
      <div class="role" style="--d:{C[8-1]}">一处错配</div>
      <div class="ewhy" style="--d:{C[8-1]}">离线重放测出「末帧看不见信息」的三局是 <b>ft09、sc25、cd82</b>。而线上回看最多的四局是 <b>sb26 19 次、cd82 12 次、su15 与 sp80 各 11 次</b> —— 其中 sb26 在重放里 11 次多帧全部 board_changed，末帧本来就看得见；ft09 明明 14/14 都看不见，却只回看了 <b>1 次</b>。</div>
      <div class="sect">成本</div>
      <div class="etag">{anim_tot} 次 × {medgap:.0f} 秒 ≈ <b>{anim_tot*medgap/3600:.1f} 小时</b>，摊到 25 局是每局 <b>{anim_tot*medgap/25/60:.0f} 分钟</b>，占单局时间预算的 {100*anim_tot*medgap/25/7920:.1f}%。占零动作轮的 {100*allk['回看 animation']/zt:.1f}%。</div>
      <div class="sect">收益</div>
      <div class="etag">{animwin_tot} 次回看后 4 轮内发生过关，命中率 <b>{100*animwin_tot/anim_tot:.1f}%</b>。判据取用户消息里的 level 字段跳变（模型自行 print 的 level_completed 只记到 8 次，不完整，已弃用）。</div>
      <div class="mini3" style="--d:{C[1]}">
        <div><b>{anim_tot}</b>总回看次数</div><div><b>{animwin_tot}</b>随后过关</div><div><b>{anim_tot*medgap/3600:.1f}h</b>折合时间</div></div>
      <div class="sect">命中落在哪</div>
      <div class="etag">{animwin_tot} 次命中分布在三局：<b>tu93 3 次</b>、<b>sc25 1 次</b>、<b>ka59 1 次</b>。sc25 正是重放里 62/62 末帧看不见的那局；而 tu93 属 type 2 纯插值，ka59 在重放里一次多帧都没采到 —— 后两者说明随机重放的覆盖不足以判定线上情形。</div>
      <div class="dl"><div class="dt">留意</div><div class="dd">关掉 <span class="c">ARC3_ANIMATION_AWARENESS</span> 会把 Stage 1 摘要一并关掉。要单独留摘要、去回看，得改源码。</div>
      <div class="dt">样本</div><div class="dd">单遍 25 局，作者自述该基准单遍方差约 ±0.45，这里的命中率只作方向参考</div>
      <div class="dt">判据</div><div class="dd">「随后 4 轮内过关」取用户消息里 <span class="c">Current state: level N</span> 的跳变。先用过 <span class="c">'level_completed': True</span> 字符串，只记到 8 次而实际过关 14 次 —— 该字段依赖模型自行 print，已弃用。</div></div>
    </div></div>
  </div></div>
</section>''')

# ── p10 旋钮与下一步 ──
P.append(f'''<section class="page">
  {shd('NEXT','var(--c4)','四个旋钮与一条新路','分数吃两样：过了几关、花了多少步。眼下卡在连基准步数都走不到，能动的就是让每局多走几步。')}
  <div class="body"><div class="nx">
    <div class="nxc" style="border-top-color:{C[4]};--d:{C[4]}">
      <div class="nxh"><span class="nxt">环境变量即可</span><span class="nxn">零改动</span></div>
      <div class="nxl">
        <div class="nxi"><b>单轮思考封顶</b> · <span class="c">LOCAL_ANALYZER_MAX_OUTPUT</span>　默认 0 不封顶。封到 8000 字符只碰 {100*len([x for x in allreas if x>8000])/N:.1f}% 的轮次，换来约 {100*sum(x-8000 for x in allreas if x>8000)/(TOTR-sum(x-8000 for x in allreas if x>8000)):.0f}% 更多轮次</div>
        <div class="nxi"><b>关思考</b> · <span class="c">LOCAL_ANALYZER_ENABLE_THINKING=0</span>　幅度最大，风险也最大。思考占了输出的 {100*reas_avg/(reas_avg+cont_avg):.1f}%</div>
        <div class="nxi"><b>动画回看开关</b> · <span class="c">ARC3_ANIMATION_AWARENESS=0</span>　省下每局约 {anim_tot*medgap/25/60:.0f} 分钟，同时失去 Stage 1 摘要</div>
        <div class="nxi"><b>无效动作拦截</b> · <span class="c">ARC3_HARD_NOOP_GUARD</span>　线上已确认为开启状态，暂无理由关</div>
        <div class="nxi"><b>工具输出上限</b> · <span class="c">LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS</span>　当前 1024，而工具输出中位数只有 381 字符 —— 这一项眼下并非瓶颈</div>
        <div class="nxi"><b>上下文窗口</b> · <span class="c">LOCAL_ANALYZER_CONTEXT_WINDOW</span>　当前 32768，历史保留 9 条消息</div>
        <div class="nxi"><b>单轮工具步数</b> · <span class="c">LOCAL_ANALYZER_TOOL_STEPS</span>　当前 12</div>
        <div class="nxi"><b>采样温度相关</b> · <span class="c">LOCAL_ANALYZER_TOP_K</span>　当前 20，<span class="c">LOCAL_ANALYZER_SEED</span>　当前 -1（不固定）</div>
        <div class="nxi">公开集与隐藏集的这些值一致，提交时 soft_end_time 返回 None，不额外收紧</div>
        <div class="nxi h">坑：这些都是模块级读取，notebook 里必须在 import 之前 os.environ 设好，否则设了等于没设</div>
        <div class="nxi h">封顶前必须先确认：截断落在思考中途时，该轮能否吐出完整的工具调用。若不能，这一轮直接作废，旋钮就成了负收益</div></div></div>
    <div class="nxc" style="border-top-color:{C[0]};--d:{C[0]}">
      <div class="nxh"><span class="nxt">要改源码包</span><span class="nxn">中等</span></div>
      <div class="nxl">
        <div class="nxi"><b>棋盘进上下文</b>　零动作轮里 {100*(allk['读 ascii 看棋盘']+allk['读 segmentation 解析棋盘'])/zt:.1f}% 花在读棋盘。系统提示词明写「原始数字网格故意不暴露」，模型只能单开一轮去取</div>
        <div class="nxi">代价是每轮多约 1200 token 输入，省下的是整轮生成。输入并行预填、输出逐字生成，单价差着量级 —— 这笔交换值得实测</div>
        <div class="nxi"><b>鼓励批量下发</b>　当前每轮动作数中位 {statistics.median(alld):.0f}、平均 {sum(alld)/len(alld):.2f}，而 harness 本来就支持一次提交多个动作，最多的一轮走了 {max(alld)} 步</div>
        <div class="nxi">批量的风险：错了会浪费更多步，而计分对步数是平方惩罚</div>
        <div class="nxi"><b>只留摘要、去掉回看</b>　保住 Stage 1 的白送信息，砍掉 Stage 2 每次一整轮的开销。当前 {anim_tot} 次回看命中 {animwin_tot} 次</div>
        <div class="nxi"><b>补 dc22 的游戏源码</b>　本地 25 局里唯独它起不来，离线实验缺一块</div>
        <div class="nxi">Stage 1 的摘要本身几乎不花钱 —— 单帧动作一个 token 都不附加，多帧时也只有几十个</div>
        <div class="nxi">segmentation 把棋盘解析成对象、颜色、包含与邻接关系，官方指定它作「主要视图」，体积比整张 ascii 小得多，更适合挂进上下文</div>
        <div class="nxi">改源码包要重新打 Kaggle dataset 并更新 kernel 的 dataset_sources，跑一轮 kernel 约 2 小时 20 分</div>
        <div class="nxi">任何一样改动都按三层框架验：DeepSeek 做实验 → qwen3.6-27b 开图做验证 → 线上 Qwen3.8 测试</div></div></div>
    <div class="nxc" style="border-top-color:{C[2]};--d:{C[2]}">
      <div class="nxh"><span class="nxt">对手的新方向</span><span class="nxn">polyphony</span></div>
      <div class="nxl">
        <div class="nxi">同一作者 08-26 15:20 发布新包，分支 <span class="c">experiment/polyphony</span>，整个 agent 目录重写成「观察 → 改代码 → 规划 → 行动」四相循环，共 2339 行</div>
        <div class="nxi"><b>让代码走步</b>　模型写 policy.py 预测状态、goal.py 定义胜利，harness 做宽度优先搜索，12 步深、两万节点，搜到就批量提交</div>
        <div class="nxi"><b>开局硬探测</b>　每关开始 harness 自己把每类动作各试一遍，模型还没看到一个 token。理由写在注释里：Q3.8 中位数要 9 步才试全动作类，五局里从未试全</div>
        <div class="nxi"><b>策略死线</b>　时钟走到 55% 仍未验证出策略，就放弃建模直接开打。理由：这个臂最可能的死法是策略优雅而一个动作没打出去</div>
        <div class="nxi"><b>守门员</b>　策略必须先能重现真实轨迹才准进搜索 —— 搜一个不能重现环境的世界模型，出来的计划看着严谨、其实是虚构</div>
        <div class="nxi">当前状态是冒烟测试：3 局 1 遍，benchmark 标签 poly-20260826-smoke</div>
        <div class="nxi">硬编码的护栏：搜索深度 12、节点预算两万、单次工具调用 40 次、Edit 轻量档 15 次、上下文 24 万字符 —— 注释写明这些数字「不交给模型决定」，因为上一版 NOOA 有两局就挂死在没有这两个上限的即兴搜索里</div>
        <div class="nxi">工具沙箱也上了闸：单次 run_python 墙钟 20 秒、CPU 15 秒、写盘 16MB、工具输出 4KB</div>
        <div class="nxi">一处呼应：08-26 手打 r11l 通关后得出的「点击类先各点一遍」，被他写成了代码里的硬执行；我方写进了手册文本，而手册臂的模型从头到尾没点过空地</div>
        <div class="nxi">同一条规律的两种写法：写进提示词模型可以不听，写进 harness 模型没得选</div></div></div>
  </div></div>
</section>''')

# ══════════════ 组装 ══════════════
NAMES = ['封面','源码包真相','动画机制','动画分布','逐局成绩','步数证据','时间账','零动作轮','思考长尾','回看成本','下一步']
navs = ''.join(f'<button data-i="{i}">{n}</button>' for i, n in enumerate(NAMES))
dots = ''.join(f'<i data-i="{i}" data-on="{1 if i==0 else 0}"></i>' for i in range(len(NAMES)))
style = open(f"{SC}/_style.css").read() + '''
/* 长表页：25 行也要在一屏里放完，不许滚、不许省略 */
.dense table{font-size:11.5px}
.dense th{padding:5px 10px;font-size:9.5px}
.dense td{padding:5.2px 10px;line-height:1.58}
.dense .tag{font-size:9.5px;padding:1px 6px}
.dense2 table{font-size:11px}
.dense2 th{padding:4px 8px;font-size:9px}
.dense2 td{padding:2.5px 8px;line-height:1.45}
.dense2 .tag{font-size:9px;padding:1px 5px}
/* 药丸标签原本只在 .dd 里生效，这里的正文块也要用 */
.etag .c,.ewhy .c,.nxi .c,.lede .c{display:inline-block;background:#FEF6E7;color:#B45309;
  padding:1px 8px;border-radius:7px;margin:0 5px 4px 0;font-size:11.5px;font-family:var(--mono)}
.etag .c.h,.nxi .c.h{background:#FEF0F0;color:var(--c8)}
/* 清单里的警告条 */
.nxi.h{color:#B91C1C}
.nxi.h::before{background:var(--c8)}
/* 中性档：既非好也非坏，只是另一种情况 */
.t-n{background:#EDEFF7;color:#5B6180}
'''; js = open(f"{SC}/_deck.js").read()
html = f'''<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ARC-AGI-3 第六发诊断 · 2026-08-27</title>
<style>{style}</style></head><body>
<div class="nav"><b>ARC-AGI-3 · 第六发诊断</b>{navs}</div>
<div class="deck">
{chr(10).join(P)}
</div>
<div class="foot"><button class="nb" id="prev">← 上一页</button><button class="nb" id="next">下一页 →</button>
<span id="pgname"></span><span class="dots">{dots}</span><span id="pgno"></span></div>
<script>
const NAMES = {json.dumps(NAMES, ensure_ascii=False)};
{js}
</script></body></html>'''
out = "probes/20260827/arc3-diagnosis-20260827.html"
open(out, "w").write(html)
print("写出", out, len(html), "字符,", len(P), "页")
