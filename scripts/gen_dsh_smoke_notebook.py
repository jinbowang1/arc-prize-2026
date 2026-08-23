"""生成 dsh 赛场冒烟 notebook (覆写 arc3-jinbo-llm-smoke.ipynb).

验证链: vLLM(Qwen3.8, tool-calling) <- dsh(离线bundle) -> game_server(3局对照)。
最大风险=Qwen 在 dsh 原生 function calling 下的表现, 只能真机验。
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

C1 = '''import os, sys, subprocess, time, json
from pathlib import Path

os.environ["ONLY_RESET_LEVELS"] = "true"
WORKING = Path("/kaggle/working")
import torch
print("GPU:", torch.cuda.get_device_name(0), "| CC:", torch.cuda.get_device_capability(0))
assert torch.cuda.get_device_capability(0) >= (8, 9), "FP8 需要 CC>=8.9"

def find_wheelhouse(pattern):
    for p in Path("/kaggle/input").rglob(pattern):
        return p.parent
    raise RuntimeError(f"找不到 {pattern}")

vllm_wheels = find_wheelhouse("vllm-*.whl")
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--no-index",
                       "--no-warn-conflicts", "--find-links", str(vllm_wheels), "vllm"])
arc_wheels = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels")
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--no-index",
                       "--no-warn-conflicts", "--find-links", str(arc_wheels), "arc-agi"])
import importlib.metadata as md_
print("vllm", md_.version("vllm"), "| arc-agi", md_.version("arc-agi"))

model_dir = None
for cfg in Path("/kaggle/input").rglob("config.json"):
    d = cfg.parent
    if list(d.glob("*.safetensors")):
        model_dir = d
        break
assert model_dir, "找不到模型目录"
bundle = next(Path("/kaggle/input").rglob("arc3-jinbo-bundle.json")).parent
sys.path.insert(0, str(bundle))
dsh_tgz = next(Path("/kaggle/input").rglob("dsh-bundle.tgz"))
print("model:", model_dir, "| src:", bundle, "| dsh:", dsh_tgz)'''

C2 = '''DSH_ROOT = Path("/kaggle/tmp/dsh")
DSH_ROOT.mkdir(parents=True, exist_ok=True)
subprocess.check_call(f"tar xzf {dsh_tgz} -C {DSH_ROOT}", shell=True)
NODE = DSH_ROOT / "node/bin/node"
DSH_BIN = DSH_ROOT / "dsh-src/apps/cli/lib/bin.js"
subprocess.check_call(f"{NODE} {DSH_BIN} --version", shell=True)
print("dsh 解包+可执行 OK", flush=True)'''

C3 = '''from kaggle_agent.serve_vllm import start_vllm
# 先试全局关思考(Qwen3 chat template kwarg); vLLM 版本不认这个参数就带思考跑
proc = None
for parser in ("qwen3_coder", "qwen3_xml", "hermes"):
    try:
        proc = start_vllm(str(model_dir), port=8000, max_model_len=32768, tool_calling=True,
                          tool_parser=parser, log_path=str(WORKING / "vllm.log"), timeout_s=600)
        print(f"vLLM up (tool parser={parser} + qwen3 reasoning parser)")
        break
    except Exception as e:
        print(f"parser={parser} 起失败:", repr(e))
assert proc, "vLLM 三种 tool parser 都起不来"'''

C4 = '''# dsh + vLLM 工具调用冒烟: 让它用 bash 工具产出文件 —— function calling 全链验证
import shutil
home = Path("/kaggle/tmp/dsh-home"); home.mkdir(parents=True, exist_ok=True)
ws = Path("/kaggle/tmp/smoke-ws"); shutil.rmtree(ws, ignore_errors=True); ws.mkdir(parents=True)
env = dict(os.environ, DSH_HOME=str(home), DEEPSEEK_API_KEY="local",
           # Kaggle 容器本身是一次性沙箱; dsh 的 Landlock/bubblewrap 在这不可用,
           # headless 的 sandbox-policy 读这个环境变量放开 bash
           DSH_PERMISSION_MODE="danger-full-access")
patch = bundle / "kaggle_agent/dsh/vllm.patch.yml"
r = subprocess.run(
    [str(NODE), str(DSH_BIN), "--profile", "headless", "--patch", str(patch),
     "用 bash 工具执行 echo smoke-ok > result.txt, 然后读出内容回复"],
    cwd=ws, env=env, capture_output=True, text=True, timeout=600)
print("dsh rc:", r.returncode)
print("stdout尾:", r.stdout[-500:])
print("stderr尾:", r.stderr[-300:])
ok = (ws / "result.txt").exists()
print("工具调用冒烟:", "PASS" if ok else "FAIL(看上方输出定责)")'''

C4B = '''# 诊断: vLLM 原始响应 + dsh 会话账本
import urllib.request
req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
    data=json.dumps({"model": "local", "max_tokens": 2000,
        "messages": [{"role": "user", "content": "用 run_bash 工具执行 echo hi"}],
        "tools": [{"type": "function", "function": {"name": "run_bash",
            "description": "run a bash command",
            "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}},
                           "required": ["cmd"]}}}]}).encode(),
    headers={"Content-Type": "application/json"})
raw = json.loads(urllib.request.urlopen(req, timeout=120).read())
msg = raw["choices"][0]["message"]
print("content:", repr((msg.get("content") or "")[:200]))
print("reasoning:", repr((msg.get("reasoning_content") or "")[:200]))
print("tool_calls:", json.dumps(msg.get("tool_calls"))[:300])
print("finish:", raw["choices"][0].get("finish_reason"))
# dsh 会话账本尾部
import glob as _g
logs = sorted(_g.glob(str(home / "sessions/**/session*.jsonl*"), recursive=True))
print("session logs:", logs[-2:])
for f in logs[-1:]:
    if f.endswith(".zstd"):
        subprocess.run(f"zstd -dc {f} | tail -c 2500", shell=True)
    else:
        print(open(f).read()[-2500:])'''

C5 = '''# 3 局对照: game_server + dsh, 每局 8 分钟墙钟
results = []
task_md = (bundle / "kaggle_agent/dsh/TASK_FULL.md").read_text()
env_dir = "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files"
for game in ["r11l", "ft09", "ls20"]:
    gs = subprocess.Popen(
        [sys.executable, "-m", "kaggle_agent.game_server", "--game", game,
         "--port", "18999", "--max-actions", "200", "--env-dir", env_dir],
        cwd=bundle, stdout=open(WORKING / f"gs_{game}.log", "w"), stderr=subprocess.STDOUT)
    time.sleep(8)
    ws_g = Path(f"/kaggle/tmp/ws-{game}"); ws_g.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        d = subprocess.run(
            [str(NODE), str(DSH_BIN), "--profile", "headless", "--patch", str(patch), task_md],
            cwd=ws_g, env=env, capture_output=True, text=True, timeout=480)
        tail = d.stdout[-400:] + " ||stderr|| " + d.stderr[-300:]
    except subprocess.TimeoutExpired as te:
        tail = f"(8分钟墙钟到, 掐掉) {str(te.stdout or '')[-200:]}"
    import urllib.request
    st = json.loads(urllib.request.urlopen("http://127.0.0.1:18999/state", timeout=10).read())
    rec = dict(game=game, level=st["level"], steps=st["steps_used"],
               seconds=round(time.time() - t0, 1))
    print(rec, flush=True)
    (WORKING / f"dsh_{game}.log").write_text(tail)
    # 会话账本诊断: dsh 中间轮次全在 session.jsonl.zstd 里, 解出尾部才知道模型每轮干了啥
    try:
        import zstandard as _zstd, glob as _g
        logs = sorted(_g.glob(str(home / "sessions/**/session*.jsonl.zstd"), recursive=True),
                      key=os.path.getmtime)
        if logs:
            with open(logs[-1], "rb") as fh:
                data = _zstd.ZstdDecompressor().stream_reader(fh).read()
            (WORKING / f"session_{game}_tail.txt").write_text(data[-12000:].decode("utf-8", "replace"))
            print(f"[{game}] 账本尾已存, 总长{len(data)}")
    except Exception as e:
        print(f"[{game}] 账本解压失败:", repr(e))
    results.append(rec)
    gs.terminate()
(WORKING / "dsh_smoke_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
print("done:", sum(r["level"] > 0 for r in results), "/ 3 局过 L1")'''


def cell(src, kind="code"):
    c = {"cell_type": kind, "metadata": {}, "source": src}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                "name": "python3"},
                 "language_info": {"name": "python", "version": "3.12"}},
    "cells": [
        cell("# dsh 赛场冒烟 — vLLM(Qwen3.8 tool-calling) + dsh 离线 bundle + 3 局对照\n\n"
             "开发用 notebook, 不是提交物。", "markdown"),
        cell(C1), cell(C2), cell(C3), cell(C4), cell(C4B), cell(C5),
    ],
}
out = ROOT / "kaggle_agent" / "notebook" / "arc3-jinbo-llm-smoke.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print("written:", out)
