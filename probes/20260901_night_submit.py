# -*- coding: utf-8 -*-
"""09-01 晚自动提交链: 规则=复刻优先(用户拍板)。
等 fn-replica 与 fn-combo 到终态 → 复刻健康(25局/零崩/有parquet)就交复刻;
复刻不健康而合体健康则交合体; 都不健康不交。
提交后必须在提交列表看到新记录才算成功(验证式提交)。
"""
import json, re, subprocess, sys, time, os

for k in ('http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY'):
    os.environ.pop(k, None)
KAGGLE = os.path.expanduser('~/.local/bin/kaggle')
PY = os.path.expanduser('~/.local/share/uv/tools/kaggle/bin/python')
COMP = 'arc-prize-2026-arc-agi-3'
CANDS = [('replica', 'jinbowang1/arc3-fnrep-run', 4)]  # v4=收尾清扫修复版; 合体v5输出已废不列
LOG = os.path.expanduser('~/Desktop/project/arc-agi-3/probes/20260901_night_submit.log')


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')


PROXY = 'http://127.0.0.1:6789'


def run(cmd, timeout=120, proxy=False):
    env = dict(os.environ)
    if proxy:
        env['http_proxy'] = env['https_proxy'] = PROXY
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, 124, '', 'timeout')


def run2(cmd, timeout=120):
    """直连失败自动换代理重试(今晚两条通道轮流抽风)。"""
    r = run(cmd, timeout=timeout)
    if r.returncode != 0:
        r2 = run(cmd, timeout=timeout, proxy=True)
        if r2.returncode == 0:
            return r2
    return r


def status(ref):
    for proxy in (False, True):
        r = run([KAGGLE, 'kernels', 'status', ref], proxy=proxy)
        m = re.search(r'KernelWorkerStatus\.([A-Z]+)', r.stdout + r.stderr)
        if m:
            return m.group(1)
    return 'UNKNOWN'


def console_stats(ref):
    code = f'''
import json, re
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
api = KaggleApi(); api.authenticate()
req = ApiListKernelSessionOutputRequest()
req.user_name, req.kernel_slug = "{ref}".split("/"); req.page_size = 1
with api.build_kaggle_client() as kc:
    resp = kc.kernels.kernels_api_client.list_kernel_session_output(req)
txt = "".join(e.get("data","") for e in json.loads(resp.log) if isinstance(e, dict))
fins = re.findall(r"\\[finished\\] \\S+ state=(\\S+) level=(\\d+)/\\d+ score=([\\d.]+)", txt)
print(json.dumps(dict(games=len(fins), crashed=sum(1 for f in fins if f[0]=="crashed"),
    opened=sum(1 for f in fins if float(f[2])>0), levels=sum(int(f[1]) for f in fins),
    mean=round(sum(float(f[2]) for f in fins)/max(1,len(fins)),2))))
'''
    r = run2([PY, '-c', code], timeout=240)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        log(f'{ref} 日志解析失败: {(r.stdout + r.stderr)[-200:]}')
        return None


def healthy(ref, name):
    if status(ref) != 'COMPLETE':
        log(f'{name} 未正常完成, 弃')
        return None
    s = console_stats(ref)
    if not s:
        return None
    log(f'{name}: {s}')
    if s['games'] < 25 or s['crashed'] > 0:
        log(f'{name} 崩溃或局数不足, 弃')
        return None
    r = run2([KAGGLE, 'kernels', 'files', ref, '--page-size', '500'], timeout=180)
    if 'submission.parquet' not in r.stdout:
        log(f'{name} 无 submission.parquet, 弃')
        return None
    return s


log(f'开始守夜(复刻优先): {CANDS}')
unknown_streak = 0
while True:
    st = {n: status(ref) for n, ref, _ in CANDS}
    log(f'状态: {st}')
    if any(s == 'UNKNOWN' for s in st.values()):
        # 网络抽风不算终态; 连续2小时查不到才放弃等待
        unknown_streak += 1
        if unknown_streak < 24:
            time.sleep(300)
            continue
        log('连续2小时查询失败, 按当前所知继续走')
    unknown_streak = 0
    if all(s in ('COMPLETE', 'ERROR', 'CANCELACKNOWLEDGED') for s in st.values()):
        break
    time.sleep(300)

chosen = None
for name, ref, ver in CANDS:
    s = healthy(ref, name)
    if s:
        chosen = (name, ref, ver, s)
        break

if not chosen:
    log('NO_CANDIDATE: 都不健康, 不提交')
    sys.exit(1)

name, ref, ver, s = chosen
log(f"选中 {name}(复刻优先规则) 开张{s['opened']}/总关{s['levels']}/均分{s['mean']} → 提交 {ref} v{ver}")
msg = f"Flash-Next 125B {name}: Son Pham serving recipe (opened {s['opened']}, levels {s['levels']}, public mean {s['mean']})"
before = run2([KAGGLE, 'competitions', 'submissions', COMP]).stdout
for attempt in range(1, 4):
    r = run2([KAGGLE, 'competitions', 'submit', COMP, '-k', ref, '-v', str(ver), '-f', 'submission.parquet', '-m', msg], timeout=300)
    log(f'提交尝试{attempt}: {(r.stdout + r.stderr).strip()[-160:]}')
    time.sleep(45)
    after = run2([KAGGLE, 'competitions', 'submissions', COMP]).stdout
    new_rows = [ln for ln in after.splitlines() if 'PENDING' in ln and ln not in before]
    if new_rows:
        log(f'SUBMITTED_OK: {new_rows[0][:150]}')
        sys.exit(0)
    log('提交列表无新记录, 重试')
    time.sleep(60)
log('SUBMIT_FAILED: 三次尝试后仍无新记录')
sys.exit(2)
