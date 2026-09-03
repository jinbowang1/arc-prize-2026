# -*- coding: utf-8 -*-
"""09-02 盯 55955743(今日原样重交的 v4)。昨夜发 55942733 已于交后 15.1h 落 ERROR。
终态时一并打印额度, 用来验证「ERROR 会不会把当天名额退回来」。
"""
import os, re, subprocess, sys, time

for k in ('http_proxy','https_proxy','HTTP_PROXY','HTTPS_PROXY'):
    os.environ.pop(k, None)
KAGGLE = os.path.expanduser('~/.local/bin/kaggle')
PY     = os.path.expanduser('~/.local/share/uv/tools/kaggle/bin/python')
COMP   = 'arc-prize-2026-arc-agi-3'
PROXY  = 'http://127.0.0.1:6789'
LOG    = os.path.expanduser('~/Desktop/project/arc-agi-3/probes/20260902_watch_submission.log')
SID    = '55955743'
EPOCH  = time.mktime(time.strptime('2026-09-02 14:00:26', '%Y-%m-%d %H:%M:%S'))

def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    open(LOG, 'a').write(line + '\n')

def run(cmd, timeout=180, proxy=False):
    env = dict(os.environ)
    if proxy:
        env['http_proxy'] = env['https_proxy'] = PROXY
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, 124, '', 'timeout')

def run2(cmd, timeout=180):
    r = run(cmd, timeout)
    if r.returncode != 0:
        r2 = run(cmd, timeout, proxy=True)
        if r2.returncode == 0:
            return r2
    return r

def table():
    for proxy in (False, True):
        r = run([KAGGLE, 'competitions', 'submissions', COMP], proxy=proxy)
        if r.returncode == 0 and 'fileName' in r.stdout:
            return r.stdout
    return None

def limits():
    code = ('from kaggle.api.kaggle_api_extended import KaggleApi\n'
            'from kagglesdk.competitions.types.competition_api_service import ApiGetSubmissionLimitsRequest\n'
            'api=KaggleApi(); api.authenticate()\n'
            'with api.build_kaggle_client() as kc:\n'
            '    r=ApiGetSubmissionLimitsRequest(); r.competition_name=%r\n'
            '    l=kc.competitions.competition_api_client.get_submission_limits(r)\n'
            'print("num_today=%%d num_allowed_now=%%d num_total=%%d" %% (l.num_today,l.num_allowed_now,l.num_total))\n' % COMP)
    r = run2([PY, '-c', code], timeout=240)
    return (r.stdout.strip().splitlines() or ['<取不到>'])[-1]

log('=' * 52)
log(f'改盯单发 {SID}(今日重交); 昨夜发 55942733 已 ERROR(交后 15.1h)')
misses = 0
while True:
    txt = table()
    if txt is None:
        misses += 1
        log(f'两条通道都查不到 (连续{misses}次), 8分钟后重试')
        if misses >= 12:
            log('WATCH_ABORT: 网络持续不通'); sys.exit(3)
        time.sleep(480); continue
    misses = 0
    row = next((ln for ln in txt.splitlines() if ln.strip().startswith(SID)), None)
    hrs = (time.time() - EPOCH) / 3600
    if row is None:
        log(f'{SID} 在列表里找不到, 8分钟后重试'); time.sleep(480); continue
    m = re.search(r'SubmissionStatus\.(\w+)', row)
    st = m.group(1) if m else '??'
    if st == 'PENDING':
        log(f'重交发 已跑 {hrs:.1f}h — 仍 PENDING')
        time.sleep(480); continue
    log(f'*** 终态 {SID}: {st} (交后 {hrs:.1f}h)')
    log('    ' + ' '.join(row.split())[:200])
    log('    落地后额度: ' + limits())
    sys.exit(0)
