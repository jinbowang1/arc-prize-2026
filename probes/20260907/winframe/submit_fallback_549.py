# -*- coding: utf-8 -*-
"""09-02 用户拍板: 原样重交 arc3-fnrep-run v4 (GPU额度已耗尽, 只能交已commit版本)。
🚨 吸取 09-01 幽灵提交教训: 重试前先按提交ID集合核对是否已落地, 不看命令返回码。
"""
import os, subprocess, sys, time

for k in ('http_proxy','https_proxy','HTTP_PROXY','HTTPS_PROXY'):
    os.environ.pop(k, None)
KAGGLE = os.path.expanduser('~/.local/bin/kaggle')
PY     = os.path.expanduser('~/.local/share/uv/tools/kaggle/bin/python')
COMP   = 'arc-prize-2026-arc-agi-3'
REF, VER = 'jinbowang1/arc3-fnrep-run', 4
PROXY  = 'http://127.0.0.1:6789'
LOG    = os.path.expanduser('~/Desktop/project/arc-agi-3/probes/20260907/winframe/submit_fallback.log')
MSG    = ('5.49 v4 bundle, unchanged (third roll of the same dice). Local A/B did not clear the winframe change, so keeping the known-good configuration for today's slot.')

def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    open(LOG, 'a').write(line + '\n')

def run(cmd, timeout=300, proxy=False):
    env = dict(os.environ)
    if proxy:
        env['http_proxy'] = env['https_proxy'] = PROXY
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, 124, '', 'timeout')

def run2(cmd, timeout=300):
    r = run(cmd, timeout)
    if r.returncode != 0:
        r2 = run(cmd, timeout, proxy=True)
        if r2.returncode == 0:
            return r2
    return r

def sub_ids():
    """返回提交ID集合; 拿不到返回 None(区别于空集)。"""
    code = ('import json\n'
            'from kaggle.api.kaggle_api_extended import KaggleApi\n'
            'from kagglesdk.competitions.types.competition_api_service import ApiListSubmissionsRequest\n'
            'api=KaggleApi(); api.authenticate()\n'
            'with api.build_kaggle_client() as kc:\n'
            '    r=ApiListSubmissionsRequest(); r.competition_name=%r; r.page=1\n'
            '    resp=kc.competitions.competition_api_client.list_submissions(r)\n'
            'print(json.dumps([int(s.ref) for s in resp.submissions]))\n' % COMP)
    r = run2([PY, '-c', code])
    try:
        import json
        return set(json.loads(r.stdout.strip().splitlines()[-1]))
    except Exception:
        log(f'  取提交列表失败: {(r.stdout+r.stderr)[-160:]}')
        return None

def limits():
    code = ('from kaggle.api.kaggle_api_extended import KaggleApi\n'
            'from kagglesdk.competitions.types.competition_api_service import ApiGetSubmissionLimitsRequest\n'
            'api=KaggleApi(); api.authenticate()\n'
            'with api.build_kaggle_client() as kc:\n'
            '    r=ApiGetSubmissionLimitsRequest(); r.competition_name=%r\n'
            '    l=kc.competitions.competition_api_client.get_submission_limits(r)\n'
            'print(l.num_today, l.num_allowed_now, l.num_total)\n' % COMP)
    r = run2([PY, '-c', code])
    try:
        return tuple(int(x) for x in r.stdout.strip().splitlines()[-1].split())
    except Exception:
        return None

log('=' * 60)
log(f'第十三发 winframe: {REF} v{VER}')
lim = limits()
log(f'交之前 limits (num_today, num_allowed_now, num_total) = {lim}')
if lim and lim[1] < 1:
    log('ABORT: 今日名额已用尽, 不交'); sys.exit(1)

before = sub_ids()
if before is None:
    log('ABORT: 交之前拿不到提交列表, 无法核对, 不冒险交'); sys.exit(2)
log(f'交之前提交数 = {len(before)}')

for attempt in (1, 2, 3):
    log(f'--- 提交尝试 {attempt}')
    r = run2([KAGGLE, 'competitions', 'submit', COMP, '-k', REF, '-v', str(VER),
              '-f', 'submission.parquet', '-m', MSG])
    out = (r.stdout + r.stderr).strip().replace('\n', ' ')
    log(f'  命令返回: rc={r.returncode} {out[-180:]}')
    time.sleep(50)
    after = sub_ids()
    if after is None:
        log('  核对时拿不到列表, 30秒后再核对(不重复提交)')
        time.sleep(30)
        after = sub_ids()
    if after is None:
        log('  仍拿不到列表; 为避免重复提交, 停在这里等人工核对'); sys.exit(3)
    new = after - before
    if new:
        log(f'SUBMITTED_OK: 新提交 {sorted(new)}')
        log(f'交之后 limits = {limits()}')
        sys.exit(0)
    log('  提交列表无新 ID, 确认没落地, 重试')
    time.sleep(45)

log('SUBMIT_FAILED: 三次尝试后仍无新记录')
sys.exit(4)
