"""在 Kaggle GPU notebook 里起 vLLM 服务并等它就绪。

用法(notebook 里):
    from kaggle_agent.serve_vllm import start_vllm
    proc = start_vllm("/kaggle/input/<模型挂载目录>", port=8000)
    os.environ["A3_LLM_BASE_URL"] = "http://127.0.0.1:8000"
    os.environ["A3_LLM_MODEL"] = "local"

⚠️模型目录直接用 Kaggle Models 挂载路径, 不复制(权重几十 GB, 拷一遍就把
磁盘配额吃光)。served-model-name 固定叫 local, agent 侧配置与模型无关。
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request


def start_vllm(
    model_dir: str,
    port: int = 8000,
    max_model_len: int = 16384,
    gpu_mem: float = 0.90,
    tool_calling: bool = False,
    extra_args: list[str] | None = None,
    timeout_s: float = 1800.0,
    log_path: str = "vllm.log",
) -> subprocess.Popen:
    """启动 vllm serve 子进程, 阻塞到 /v1/models 就绪(权重加载可能要十几分钟)。

    tool_calling: dsh 这类靠原生 function calling 的 agent 需要 vLLM 开启
    工具调用解析(Qwen3 系用 hermes parser); REPL 范式(模型只回代码块)不用开。"""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_dir,
        "--served-model-name", "local",
        "--port", str(port),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_mem),
    ]
    if tool_calling:
        cmd += ["--enable-auto-tool-choice", "--tool-call-parser", "hermes"]
    cmd += extra_args or []
    log = open(log_path, "w")  # noqa: SIM115  (进程生命周期同 notebook, 不关)
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)

    probe = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vllm 进程退出(code={proc.returncode}), 看 {log_path}")
        try:
            with urllib.request.urlopen(probe, timeout=5) as r:
                json.loads(r.read())
            print(f"vLLM ready on :{port}")
            return proc
        except Exception:  # noqa: BLE001
            time.sleep(10)
    proc.terminate()
    raise RuntimeError(f"vLLM {timeout_s}s 内未就绪, 看 {log_path}")
