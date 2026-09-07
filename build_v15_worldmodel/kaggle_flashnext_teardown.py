from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path


working = Path(os.environ["TAAF_KAGGLE_WORKING_DIR"])
results: dict[str, str] = {}
for label, filename in (
    ("curator", "world-model-curator/curator.pid"),
    ("vllm", "vllm.pid"),
):
    pid_path = working / filename
    if not pid_path.is_file():
        results[label] = "pid-file-missing"
        continue
    try:
        pid = int(pid_path.read_text().strip())
        os.killpg(pid, signal.SIGTERM)
        results[label] = f"terminated-process-group-{pid}"
    except (OSError, ValueError) as exc:
        results[label] = f"already-stopped-or-invalid: {exc}"
time.sleep(3)
(working / "flashnext-teardown.json").write_text(
    json.dumps(results, indent=2) + "\n", encoding="utf-8"
)
print("Flash-Next teardown:", results, flush=True)

# 09-01 修复4: 收尾必须删掉 /kaggle/working 里的大件(解包运行时~15GB+模型软链目录),
# 否则 Kaggle 保存输出时被撑死, 整个输出作废(v3/v5 输出只剩824字节空壳, parquet 丢失)。
import shutil

for name in ("flashnext-gcp-runtime", "flashnext-preconverted-model"):
    target = working / name
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
        print(f"Flash-Next teardown: removed {target}", flush=True)
log_path = working / "vllm.log"
if log_path.is_file():
    tail_lines = log_path.read_text(errors="replace").splitlines()[-4000:]
    (working / "vllm-tail.log").write_text("\n".join(tail_lines), encoding="utf-8")
    log_path.unlink()
    print("Flash-Next teardown: vllm.log -> vllm-tail.log", flush=True)

