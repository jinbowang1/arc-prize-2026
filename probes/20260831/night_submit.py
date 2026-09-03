# -*- coding: utf-8 -*-
"""08-31 晚自动提交链: 等两个修正文案 kernel 跑完 → 按用户拍板规则(开张局数高者)选一个 →
提交 → 验证提交真的挂上(提交列表出现新记录才算数, 上周的教训)。
用法: night_submit.py <manual版本号> <both版本号>
全程写 night_submit.log; 终局输出 SUBMITTED_OK / SUBMIT_FAILED / NO_CANDIDATE。
"""
import json, re, subprocess, sys, time, os

os.environ.pop('http_proxy', None); os.environ.pop('https_proxy', None)
os.environ.pop('HTTP_PROXY', None); os.environ.pop('HTTPS_PROXY', None)

KAGGLE = os.path.expanduser('~/.local/bin/kaggle')
PY = os.path.expanduser('~/.local/share/uv/tools/kaggle/bin/python')
COMP = 'arc-prize-2026-arc-agi-3'
# 参数: <manual版本号> <合并版slug> <合并版版本号>  (08-31晚: both名字环境脏了换combo)
KERNELS = {'manual': ('jinbowang1/arc3-l1-manual', int(sys.argv[1])),
           'combo': (f'jinbowang1/{sys.argv[2]}', int(sys.argv[3]))}
LOG = os.path.expanduser('~/Desktop/project/arc-agi-3/probes/20260831/night_submit.log')


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')


def run(cmd, timeout=120):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def status(ref):
    r = run([KAGGLE, 'kernels', 'status', ref])
    m = re.search(r'KernelWorkerStatus\.([A-Z]+)', r.stdout + r.stderr)
    return m.group(1) if m else 'UNKNOWN'


def console_stats(ref):
    """开张/总关/均分/崩溃数, 取自 kernel 控制台日志的 [finished] 行。"""
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
out = dict(games=len(fins), crashed=sum(1 for f in fins if f[0]=="crashed"),
           opened=sum(1 for f in fins if float(f[2])>0), levels=sum(int(f[1]) for f in fins),
           mean=round(sum(float(f[2]) for f in fins)/max(1,len(fins)),2))
print(json.dumps(out))
'''
    r = run([PY, '-c', code], timeout=180)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        log(f'{ref} 日志解析失败: {r.stdout[-200:]} {r.stderr[-200:]}')
        return None


def has_parquet(ref):
    r = run([KAGGLE, 'kernels', 'files', ref, '--page-size', '500'], timeout=180)
    return 'submission.parquet' in r.stdout


# 1. 等两个 kernel 到终态
log(f'开始守夜: {KERNELS}')
while True:
    st = {k: status(ref) for k, (ref, _) in KERNELS.items()}
    log(f'状态: {st}')
    if all(s in ('COMPLETE', 'ERROR', 'CANCELACKNOWLEDGED', 'UNKNOWN') for s in st.values()):
        break
    time.sleep(300)

# 2. 汇总候选
cands = []
for k, (ref, ver) in KERNELS.items():
    if status(ref) != 'COMPLETE':
        log(f'{k} 未正常完成, 弃')
        continue
    s = console_stats(ref)
    if not s:
        continue
    log(f'{k}: {s}')
    if s['games'] < 25 or s['crashed'] > 0:
        log(f'{k} 有崩溃或局数不足, 弃')
        continue
    if not has_parquet(ref):
        log(f'{k} 没有 submission.parquet, 弃')
        continue
    cands.append((s['opened'], s['levels'], s['mean'], k, ref, ver))

if not cands:
    log('NO_CANDIDATE: 两个都不合格, 不提交')
    sys.exit(1)

cands.sort(reverse=True)  # 开张 > 总关 > 均分
opened, levels, mean, k, ref, ver = cands[0]
log(f'选中 {k} (开张{opened}/总关{levels}/均分{mean}) → 提交 {ref} v{ver}')

# 3. 提交 + 验证(最多3次)
msg = f'L1 {k} fixed-wording: V31 stack + first-level patches (opened {opened}, levels {levels}, public mean {mean})'
before = run([KAGGLE, 'competitions', 'submissions', COMP]).stdout
for attempt in range(1, 4):
    r = run([KAGGLE, 'competitions', 'submit', COMP, '-k', ref, '-v', str(ver), '-f', 'submission.parquet', '-m', msg], timeout=300)
    log(f'提交尝试{attempt}: {(r.stdout + r.stderr).strip()[-200:]}')
    time.sleep(45)
    after = run([KAGGLE, 'competitions', 'submissions', COMP]).stdout
    new_rows = [ln for ln in after.splitlines() if 'PENDING' in ln and ln not in before]
    if new_rows:
        log(f'SUBMITTED_OK: 提交列表已出现新记录: {new_rows[0][:160]}')
        sys.exit(0)
    log('提交列表没看到新记录, 重试')
    time.sleep(60)
log('SUBMIT_FAILED: 三次尝试后提交列表仍无新记录')
sys.exit(2)
