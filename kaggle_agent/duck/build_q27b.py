"""09-18 换 27B (arc3-q27b): 5.49 的 harness 不动, 只把伺服栈换成 Tufa 09-01 公开配方
(vLLM 0.19 wheelhouse + Qwen3.8-27B-FP8, --max-model-len 65536, 无 MTP/无 FP8 KV/无 async)。

## 为什么
Flash-Next 125B NVFP4 权重 ~62 GB, 只给 KV 留 14.09 GiB = 488,898 tokens, 32k 上下文只够 14.92 路并发,
而我们开 28 路(超配 1.9 倍) -> `Waiting: 7`, 四分之一的局一直排队干等, 它们的 2.2h 闸刀照样在走。
27B FP8 权重 ~27 GB -> KV 约 54 GB(4 倍), 28 路 x 64k = 1.79M tokens 刚好装下 —— Tufa 的 concurrency 28 +
max-model-len 65536 就是卡着这个上限配的。赌的是 KV 收益 > 模型能力损失(他们 09-01 同配方 4.71, 我们 125B 5.49)。

## 怎么改(只动伺服, 线上 harness 数据集一个字不动)
- 数据集: fnrep-source(我们的 harness) + driessmit1/arc3-vllm-h100-wheelhouse-v3 + jakobbrggen/qwen3-8-27b-fp8-hf-snapshot
- cell 8: 不再执行 bundle 里 Son 的 kaggle_flashnext_preconverted_setup.py, 改为逐字执行 Tufa v27 的 PYSETUP
  (reference/avo/tufa-avo-v2-reimplementation/setup_commands.json), 它把 LOCAL_ANALYZER_* 写进 TAAF_KAGGLE_SETUP_ENV
- V31 看门狗整段删掉(它只认 Flash-Next 的 provenance/MTP 参数, 09-14 已查实必失败); 留空壳函数给 cell 14 调用
- cell 18: 改成打印 vLLM 日志里的真实 KV 容量 —— 跑起来第一件事看这个, 远低于 1.8M tokens 就说明账算错了

用法: python3 build_q27b.py -> kernels/k_q27b
"""
from __future__ import annotations
import copy, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE_NB = HERE / "kernels/k_fnrep_run/arc3-fnrep-run.ipynb"
BASE_META = HERE / "kernels/k_fnrep_run/kernel-metadata.json"
TUFA_SETUP = ROOT / "reference/avo/tufa-avo-v2-reimplementation/setup_commands.json"
SLUG = "arc3-q27b"
DATASETS = ["jinbowang1/arc3-fnrep-source",
            "driessmit1/arc3-vllm-h100-wheelhouse-v3",
            "jakobbrggen/qwen3-8-27b-fp8-hf-snapshot"]


def tufa_pysetup() -> str:
    """把 Tufa setup_commands.json 里 `"$PYTHON" - <<'PYSETUP' ... PYSETUP` 的脚本体抠出来。"""
    cmds = json.loads(TUFA_SETUP.read_text(encoding="utf-8"))
    assert len(cmds) == 1 and cmds[0].startswith('"$PYTHON" - <<\'PYSETUP\''), "Tufa setup 形状变了"
    body = cmds[0].split("<<'PYSETUP'\n", 1)[1]
    body = body.rsplit("\nPYSETUP", 1)[0]
    assert "qwen3-8-27b-fp8-hf-snapshot" in body and "arc3-vllm-h100-wheelhouse-v3" in body
    assert "'--max-model-len',\n        str(VLLM_MAX_MODEL_LEN)" in body and "VLLM_MAX_MODEL_LEN = 65536" in body
    return body


CELL8_HEAD = '''# Each bundled repo exposes its importable tree at <repo>/src or <repo>.
def _source_path_entries(bundle_dir: Path) -> list:
    entries = []
    for repo in sorted((bundle_dir / "src").iterdir(), reverse=True):
        for candidate in (repo / "src", repo):
            if candidate.is_dir():
                entries.append(candidate)
    return entries


# Environment handed to each setup command (paths + any keys it has persisted).
def _command_env() -> dict:
    env = os.environ.copy()
    env["PYTHON"] = sys.executable
    env["TAAF_KAGGLE_BUNDLE_DIR"] = str(BUNDLE_DIR)
    env["TAAF_KAGGLE_WORKING_DIR"] = str(WORKING_DIR)
    env["TAAF_KAGGLE_SETUP_ENV"] = str(SETUP_ENV_PATH)
    env.update({str(k): str(v) for k, v in json.loads(SETUP_ENV_PATH.read_text()).items()})
    return env


# Make the bundled repos importable here (sys.path) and in child processes (.pth).
source_entries = _source_path_entries(BUNDLE_DIR)
for entry in source_entries:
    sys.path.insert(0, str(entry))
pth_path = Path(sysconfig.get_paths()["purelib"]) / "taaf_kaggle_sources.pth"
pth_path.write_text("".join(f"{entry}\\n" for entry in source_entries))
print(f"taaf.kaggle: wrote {pth_path} ({len(source_entries)} source roots)")


# ---- 09-18 Q27B: 伺服栈换成 Tufa 09-01 公开配方 (vLLM 0.19 wheelhouse + Qwen3.8-27B-FP8, 64k, 无 MTP/FP8KV/async) ----
# bundle 里 Son 的 kaggle_flashnext_preconverted_setup.py 不再执行; 下面这段脚本逐字来自
# jakobbrggen/taaf-kaggle-source v27 的 setup_commands.json, 它会把 LOCAL_ANALYZER_* 写进 TAAF_KAGGLE_SETUP_ENV。
Q27B_SETUP = r"""
'''

CELL8_TAIL = '''"""

env = _command_env()
print("Q27B: starting Tufa serving stack (Qwen3.8-27B-FP8 on vLLM 0.19, max-model-len 65536)", flush=True)
subprocess.run([sys.executable, "-"], input=Q27B_SETUP, text=True, check=True, cwd=WORKING_DIR, env=env)
env = _command_env()
os.environ.update(env)

# Honour any PYTHONPATH a setup command exported.
for entry in reversed([e for e in os.environ.get("PYTHONPATH", "").split(os.pathsep) if e]):
    if entry not in sys.path:
        sys.path.insert(0, entry)

# V31 看门狗只认 Flash-Next 的 provenance/MTP 参数, 09-14 查实在这条栈上必失败; Tufa 自己也不带看门狗。留空壳给 cell 14。
import urllib.request as _q27b_urllib

_Q27B_URL = os.environ.get("LOCAL_ANALYZER_BASE_URL", "http://127.0.0.1:1234/v1").rstrip("/")


def _q27b_healthy(timeout: float = 4.0) -> bool:
    try:
        with _q27b_urllib.urlopen(f"{_Q27B_URL}/models", timeout=timeout) as response:
            return 200 <= int(response.status) < 500
    except Exception:
        return False


def _v31_start_watchdog(solver) -> None:
    print("Q27B: no watchdog (Tufa serving stack runs bare).", flush=True)


def _v31_stop_watchdog() -> None:
    pass


if not _q27b_healthy(10):
    raise RuntimeError("Q27B preflight: local vLLM API is not healthy.")
print("Q27B preflight: local vLLM API healthy.", flush=True)
print("Q27B analyzer model:", os.environ.get("INFERENCE_ANALYZER_MODEL"),
      "context:", os.environ.get("LOCAL_ANALYZER_CONTEXT_WINDOW"),
      "upscale:", os.environ.get("MULTIMODAL_UPSCALE"),
      "temperature:", os.environ.get("LOCAL_ANALYZER_TEMPERATURE"), flush=True)
'''

CELL18 = '''
# Q27B validation: 真实 KV 容量是这一发的全部赌注 —— 远低于 1.8M tokens 就说明显存账算错了。
import re as _q27b_re

print("\\nQ27B VALIDATION")
print("=" * 76)
_log = WORKING_DIR / "vllm-openai-server.log"
if _log.exists():
    for line in _log.read_text(encoding="utf-8", errors="replace").splitlines():
        if _q27b_re.search(r"KV cache|Maximum concurrency|max_model_len|max-model-len|Loading weights took|Model loading took", line):
            print(line.strip()[-200:])
else:
    print("vllm-openai-server.log not found")
print("concurrency:", getattr(bm.solver, "concurrency", None), "max_runtime_s_per_game:", getattr(bm.solver, "max_runtime_s_per_game", None))
'''


def _set_source(cell, text):
    cell["source"] = text.splitlines(keepends=True)


def build(slug=SLUG):
    nb = json.loads(BASE_NB.read_text(encoding="utf-8"))
    base_nb = json.loads(BASE_NB.read_text(encoding="utf-8"))
    meta = json.loads(BASE_META.read_text(encoding="utf-8"))
    cells = nb["cells"]
    # cell 0 标题
    _set_source(cells[0], "# arc3-q27b\n\n5.49 harness (Son's Flash-Next replica bundle) with the serving stack swapped to "
                          "Tufa's 2026-09-01 public recipe: Qwen3.8-27B-FP8 on vLLM 0.19, max-model-len 65536, no MTP / FP8 KV / async. "
                          "Only cell 8 (serving) and the validation cell change; the harness dataset is byte-identical.\n")
    # cell 6 数据集
    c6 = "".join(cells[6]["source"])
    old = 'DATASET_SOURCES = ["jinbowang1/arc3-fnrep-source", "sonphamorg/arc3-flashnext-serving-part-a-v1", "sonphamorg/arc3-flashnext-serving-part-b-v1", "sonphamorg/arc3-flashnext-serving-part-c-v1", "sonphamorg/arc3-flashnext-gcp-runtime-exact-v1"]'
    assert old in c6, "cell 6 锚点"
    _set_source(cells[6], c6.replace(old, "DATASET_SOURCES = " + json.dumps(DATASETS)))
    # cell 8 伺服
    body = tufa_pysetup()
    assert '"""' not in body, "PYSETUP 里有三引号, 嵌不进去"
    _set_source(cells[8], CELL8_HEAD + body + CELL8_TAIL)
    # cell 17/18 验证
    _set_source(cells[17], "## Q27B validation\nPrints the real KV-cache capacity vLLM reported; diagnostic only.\n")
    _set_source(cells[18], CELL18)
    for i, (new, old_c) in enumerate(zip(cells, base_nb["cells"])):
        if i in (0, 6, 8, 17, 18): continue
        assert "".join(new["source"]) == "".join(old_c["source"]), f"cell {i} 不该动"
    for cell in cells:
        if cell["cell_type"] == "code":
            cell["outputs"] = []; cell["execution_count"] = None
    out = HERE / f"kernels/k_{slug.replace('arc3-', '')}"
    out.mkdir(parents=True, exist_ok=True)
    m = copy.deepcopy(meta); m["id"] = f"jinbowang1/{slug}"; m.pop("id_no", None)
    m["title"] = slug; m["code_file"] = f"{slug}.ipynb"; m["dataset_sources"] = list(DATASETS)
    (out / "kernel-metadata.json").write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n")
    (out / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"built {out / f'{slug}.ipynb'}")
    return out


if __name__ == "__main__":
    build()
