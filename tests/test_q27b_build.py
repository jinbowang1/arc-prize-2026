"""arc3-q27b 打包验收: 伺服栈换成 Tufa 27B 配方, harness 与其余 cell 一字不动。

验七件:
  · 数据集 = fnrep-source + wheelhouse-v3 + qwen3-8-27b 快照, Flash-Next 四个数据集全部摘掉
  · cell 8 不再引用 provenance / MTP / FP8 KV / async / Flash-Next 的 setup 脚本
  · cell 8 内嵌的 Tufa 脚本逐字等于 reference 里 setup_commands.json 的脚本体
  · vLLM 参数: 27B 快照 + max-model-len 65536 + qwen3_coder + preserve_thinking, 没有 kv-cache-dtype/speculative/async
  · cell 14 仍能找到 _v31_start_watchdog/_v31_stop_watchdog(空壳)
  · 所有 code cell 能 ast.parse; 其余 cell 与 5.49 基线逐字相同
  · 运行级: 用 cell 8 同样的调用方式(python - <<script, 同一套 TAAF_KAGGLE_* 环境)真跑内嵌脚本,
    在本机没有数据集时必须走到「Missing attached dataset path」这一步(证明环境接线通、脚本可执行)
"""
import ast, json, os, subprocess, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle_agent/duck"))
import build_q27b as B

R = []
def check(name, cond, extra=""):
    R.append((name, bool(cond))); print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  '+extra) if extra else ''}")

out = B.build()
nb = json.loads((out / "arc3-q27b.ipynb").read_text(encoding="utf-8"))
meta = json.loads((out / "kernel-metadata.json").read_text(encoding="utf-8"))
base = json.loads(B.BASE_NB.read_text(encoding="utf-8"))
cells = [("".join(c["source"]), c["cell_type"]) for c in nb["cells"]]
c6, c8, c14, c18 = cells[6][0], cells[8][0], cells[14][0], cells[18][0]

check("数据集三件", meta["dataset_sources"] == B.DATASETS, str(meta["dataset_sources"]))
check("Flash-Next 数据集全摘", not any("flashnext" in d for d in meta["dataset_sources"]) and "flashnext" not in c6.lower().split("dataset_sources = ")[1].split("]")[0])
check("cell 6 DATASET_SOURCES 与 metadata 一致", json.dumps(B.DATASETS) in c6)
c8_code = "\n".join(l for l in c8.splitlines() if not l.lstrip().startswith("#"))   # 只查代码行, 注释里提一嘴不算
for kw in ("provenance", "MTP", "mtp", "kv-cache-dtype", "speculative", "async-scheduling", "kaggle_flashnext", "_v31_spawn", "_v31_recover"):
    check(f"cell 8 代码无 {kw}", kw not in c8_code)
body = B.tufa_pysetup()
check("内嵌脚本逐字 = Tufa setup_commands 脚本体", body in c8, f"{len(body)} 字")
for kw in ("MODEL_SLUG = 'qwen3-8-27b-fp8-hf-snapshot'", "WHEELHOUSE_SLUG = 'arc3-vllm-h100-wheelhouse-v3'",
           "VLLM_MAX_MODEL_LEN = 65536", "'qwen3_coder'", '{"preserve_thinking": true}', "'--enable-prefix-caching'",
           "'LOCAL_ANALYZER_CONTEXT_WINDOW': str(ANALYZER_CONTEXT_WINDOW)", "ANALYZER_CONTEXT_WINDOW = 32768"):
    check(f"vLLM/env 参数含 {kw[:40]}", kw in body)
check("cell 14 调用的看门狗函数在 cell 8 有空壳", "_v31_start_watchdog(bm.solver)" in c14 and "def _v31_start_watchdog" in c8 and "def _v31_stop_watchdog" in c8)
check("cell 8 有健康预检", "_q27b_healthy(10)" in c8)
check("cell 18 打印 KV 容量", "KV cache" in c18 and "Maximum concurrency" in c18)
ok = True
for i, (src, typ) in enumerate(cells):
    if typ != "code": continue
    try: ast.parse(src)
    except SyntaxError as e: ok = False; print("   语法错 cell", i, e)
check("所有 code cell 可解析", ok)
same = all("".join(nb["cells"][i]["source"]) == "".join(base["cells"][i]["source"]) for i in range(len(cells)) if i not in (0, 6, 8, 17, 18))
check("其余 cell 与 5.49 基线逐字相同", same)
check("code cell 输出已清空", all(c.get("outputs") == [] for c in nb["cells"] if c["cell_type"] == "code"))

# 运行级: 同 cell 8 的调用方式跑内嵌脚本
with tempfile.TemporaryDirectory() as td:
    wd = Path(td); setup_env = wd / "taaf_setup_env.json"
    setup_env.write_text(json.dumps({"TAAF_KAGGLE_INPUT_PATHS": json.dumps({d: f"/kaggle/input/{d.split('/')[1]}" for d in B.DATASETS}),
                                     "TAAF_KAGGLE_DATASET_SOURCES": json.dumps(B.DATASETS)}))
    env = os.environ.copy()
    env.update({"PYTHON": sys.executable, "TAAF_KAGGLE_WORKING_DIR": str(wd), "TAAF_KAGGLE_SETUP_ENV": str(setup_env),
                "TAAF_KAGGLE_BUNDLE_DIR": str(wd)})
    env.update(json.loads(setup_env.read_text()))
    q = c8.split('Q27B_SETUP = r"""\n', 1)[1].split('\n"""', 1)[0]
    p = subprocess.run([sys.executable, "-"], input=q, text=True, capture_output=True, cwd=wd, env=env)
    tail = (p.stdout + p.stderr)[-600:]
    check("内嵌脚本真跑到「缺数据集」这一步(环境接线通)", p.returncode != 0 and "Missing attached dataset path" in tail, tail.strip().splitlines()[-1][:120] if tail.strip() else "")
    check("跑之前先打印了两条路径", "vLLM wheelhouse path:" in p.stdout and "Qwen model path:" in p.stdout)

n = sum(1 for _, ok_ in R if ok_)
print(f"\n{n}/{len(R)} 项通过")
sys.exit(0 if n == len(R) else 1)
