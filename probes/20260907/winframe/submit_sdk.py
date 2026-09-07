# -*- coding: utf-8 -*-
"""用 kagglesdk 直接提交(CLI 的 -f 不会传成 file_name, 会被 400 拒)。
保留 09-02 的防幽灵提交逻辑: 只信提交 ID 集合的变化, 不信命令返回码。
"""
import os, sys, time, json

os.environ.setdefault('http_proxy', 'http://127.0.0.1:6789')
os.environ.setdefault('https_proxy', 'http://127.0.0.1:6789')

COMP = 'arc-prize-2026-arc-agi-3'
OWNER = 'jinbowang1'
SLUG = os.environ.get('WF_SLUG', 'arc3-v13-winframe-run')
VER = int(os.environ.get('WF_VER', '2'))
FILE_NAME = os.environ.get('WF_FILE', 'submission.parquet')
MSG = os.environ.get('WF_MSG') or (
    'winframe fix on the 5.49 v4 base: on a level transition arcengine returns '
    '[solved board, next-level board] and GameState.frame kept only the last, so the board at '
    'the moment of the win never reached the model (measured 87% of cells differ). The solved '
    'board is now passed in the user prompt as ASCII (plus a second image when multimodal is on). '
    'Single change; no other diff from the 5.49 bundle.'
)
LOG = os.path.expanduser('~/Desktop/project/arc-agi-3/probes/20260907/winframe/submit.log')

def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOG, 'a') as f: f.write(line + '\n')

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import (
    ApiCreateCodeSubmissionRequest, ApiListSubmissionsRequest)

api = KaggleApi(); api.authenticate()

def sub_ids():
    try:
        with api.build_kaggle_client() as kc:
            r = ApiListSubmissionsRequest(); r.competition_name = COMP; r.page = 1
            resp = kc.competitions.competition_api_client.list_submissions(r)
            return {int(s.ref) for s in resp.submissions}
    except Exception as e:
        log(f'  列提交失败: {type(e).__name__} {str(e)[:120]}')
        return None

log('=' * 60)
log(f'第十三发 winframe: {OWNER}/{SLUG} v{VER}  file={FILE_NAME}')
before = sub_ids()
if before is None:
    log('ABORT: 交之前拿不到提交列表, 不冒险交'); sys.exit(2)
log(f'交之前提交数 = {len(before)}')

for attempt in (1, 2, 3):
    log(f'--- 提交尝试 {attempt}')
    try:
        with api.build_kaggle_client() as kc:
            r = ApiCreateCodeSubmissionRequest()
            r.competition_name = COMP
            r.kernel_owner = OWNER
            r.kernel_slug = SLUG
            r.kernel_version = VER
            r.file_name = FILE_NAME
            r.submission_description = MSG
            resp = kc.competitions.competition_api_client.create_code_submission(r)
        log(f'  提交调用返回: {str(resp)[:200]}')
    except Exception as e:
        body = ''
        rr = getattr(e, 'response', None)
        if rr is not None:
            try: body = json.dumps(rr.json(), ensure_ascii=False)[:300]
            except Exception: body = getattr(rr, 'text', '')[:300]
        log(f'  异常: {type(e).__name__} {str(e)[:150]} {body}')
    time.sleep(45)
    after = sub_ids()
    if after is None:
        time.sleep(30); after = sub_ids()
    if after is None:
        log('  拿不到列表, 停在这里等人工核对, 不重复提交'); sys.exit(3)
    new = after - before
    if new:
        log(f'SUBMITTED_OK: 新提交 {sorted(new)}')
        sys.exit(0)
    log('  提交列表无新 ID, 确认没落地')
    time.sleep(30)
log('SUBMIT_FAILED: 三次尝试后仍无新记录')
sys.exit(4)
