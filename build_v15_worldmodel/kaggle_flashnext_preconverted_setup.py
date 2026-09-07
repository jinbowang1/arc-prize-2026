"""Launch the exact Flash-Next candidate from an immutable preconverted model.

This is a packaging-only wrapper around ``kaggle_flashnext_setup.py``.  It
reuses the exact runtime, vLLM command, curator, and solver environment while
replacing only chunk reconstruction and PLE conversion with verification of a
pinned, already-converted Kaggle model variation.
"""

from __future__ import annotations

from pathlib import Path


bundle_dir = Path(__file__).resolve().parent
base_path = bundle_dir / "kaggle_flashnext_setup.py"
base = base_path.read_text(encoding="utf-8")

start = base.index("def resolve_source_model() -> Path:\n")
end = base.index("\ndef request_json(", start)

replacement = r'''def resolve_source_model() -> Path:
    markers = sorted(
        path.resolve()
        for path in Path("/kaggle/input").rglob(
            "PRECONVERTED_MODEL_PROVENANCE.json"
        )
        if path.is_file()
    )
    if len(markers) != 1:
        raise RuntimeError(
            "Expected exactly one immutable preconverted Flash-Next model, "
            f"got {markers}"
        )

    provenance_path = markers[0]
    # 09-01 patch: 数据集布局下三块权重分片(serving-part-000/001/002)分居三个
    # dataset 挂载点, 原逻辑只认 provenance 所在目录导致校验必失败("file drifted").
    # 永远跨 /kaggle/input 全域收集分片目录并软链接拼成一个目录, 零字节拷贝。
    shard_dirs = sorted(
        path
        for path in Path("/kaggle/input").rglob("serving-part-*")
        if path.is_dir() and path.name.startswith("serving-part-")
    )
    if len(shard_dirs) != 3:
        raise RuntimeError(
            f"Expected three serving shard dirs across mounts, got {shard_dirs}"
        )
    linked = WORKING_DIR / "flashnext-preconverted-model"
    linked.mkdir(parents=True, exist_ok=True)
    observed_names = set()
    for shard in shard_dirs:
        for mounted in sorted(shard.iterdir()):
            if not mounted.is_file():
                raise RuntimeError(f"Unexpected nested entry in serving shard: {mounted}")
            if mounted.name in observed_names:
                raise RuntimeError(f"Duplicate file across serving shards: {mounted.name}")
            observed_names.add(mounted.name)
            (linked / mounted.name).symlink_to(mounted)
    source = linked
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("format") != (
        "arc3-flashnext-preconverted-serving-model-provenance-v1"
    ):
        raise RuntimeError("Unexpected preconverted model provenance format")
    if provenance.get("model_id") != MODEL_ID:
        raise RuntimeError("Preconverted model id drifted")
    if provenance.get("revision") != MODEL_REVISION:
        raise RuntimeError("Preconverted model revision drifted")
    if int(provenance.get("source_safetensors_bytes", -1)) != (
        EXPECTED_SAFETENSORS_BYTES
    ):
        raise RuntimeError("Preconverted source byte provenance drifted")

    manifest_path = source / "PRECONVERTED_MODEL_FILE_MANIFEST.json"
    if sha256(manifest_path) != provenance.get("file_manifest_sha256"):
        raise RuntimeError("Preconverted file-manifest hash drifted")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model_id") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise RuntimeError("Preconverted file manifest identity drifted")
    files = list(manifest.get("files", []))
    if len(files) != int(provenance.get("file_count", -1)):
        raise RuntimeError("Preconverted file count drifted")
    if sum(int(item["bytes"]) for item in files) != int(
        provenance.get("total_bytes", -1)
    ):
        raise RuntimeError("Preconverted manifest total bytes drifted")
    # The original offline audit also recorded Hugging Face's download cache
    # receipts.  Kaggle's direct-file model uploader intentionally skips the
    # nested ``.cache`` directory; those receipts are neither model payload nor
    # runtime dependencies.  Pin their exact audited shape, then verify every
    # serving-payload record that Kaggle does mount.
    cache_receipts = [
        item for item in files if str(item["path"]).startswith(".cache/")
    ]
    if len(cache_receipts) != 840 or sum(
        int(item["bytes"]) for item in cache_receipts
    ) != 47_407:
        raise RuntimeError("Preconverted cache-receipt provenance drifted")
    reserved_dataset_metadata = [
        item for item in files if str(item["path"]) == "dataset-metadata.json"
    ]
    if len(reserved_dataset_metadata) != 1 or int(
        reserved_dataset_metadata[0]["bytes"]
    ) != 418:
        raise RuntimeError("Preconverted dataset-metadata provenance drifted")
    serving_files = [
        item
        for item in files
        if not str(item["path"]).startswith(".cache/")
        and str(item["path"]) != "dataset-metadata.json"
    ]
    for item in serving_files:
        path = source / str(item["path"])
        if not path.is_file() or path.stat().st_size != int(item["bytes"]):
            raise RuntimeError(f"Preconverted mounted file drifted: {path}")

    expected = json.loads(
        (BUNDLE_DIR / "FLASHNEXT_GCP_MODEL_INFO.json").read_text()
    )["ple_conversion"]
    observed = json.loads((source / "ple-bf16-conversion.json").read_text())
    expected_files = {
        item["target"]: (int(item["target_bytes"]), item["target_sha256"])
        for item in expected["files"]
    }
    observed_files = {
        item["target"]: (int(item["target_bytes"]), item["target_sha256"])
        for item in observed["files"]
    }
    if observed_files != expected_files:
        raise RuntimeError("Preconverted PLE file hashes differ from the GCP winner")
    if observed.get("index_sha256") != expected.get("index_sha256"):
        raise RuntimeError("Preconverted PLE index provenance differs from the GCP winner")
    if sha256(source / "model.safetensors.index.json") != expected["index_sha256"]:
        raise RuntimeError("Mounted preconverted index hash drifted")
    if list(source.glob("model-plefp8-*.safetensors")):
        raise RuntimeError("Mounted serving model still contains FP8 PLE source files")
    for name, (expected_bytes, _) in expected_files.items():
        target = source / name
        if not target.is_file() or target.stat().st_size != expected_bytes:
            raise RuntimeError(f"Mounted preconverted PLE file drifted: {target}")

    print(
        "Exact immutable preconverted Flash-Next model verified: "
        f"{source} ({len(serving_files)} serving files, "
        f"{sum(int(item['bytes']) for item in serving_files)} bytes)",
        flush=True,
    )
    return source


def materialize_and_convert_model(source: Path) -> None:
    global MODEL_DIR
    MODEL_DIR = source.resolve()
    observed = json.loads((MODEL_DIR / "ple-bf16-conversion.json").read_text())
    provenance = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "source_checkpoint_bytes": EXPECTED_UPSTREAM_CHECKPOINT_BYTES,
        "source_safetensors_bytes": EXPECTED_SAFETENSORS_BYTES,
        "conversion": observed,
        "serving_model": str(MODEL_DIR),
        "preconverted_immutable_input": True,
        "routed_experts": "unchanged immutable Kaggle model files",
        "container": EXPECTED_CONTAINER,
    }
    (WORKING_DIR / "flashnext-model-provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
'''

patched = base[:start] + replacement + base[end:]
if patched == base or "materialize_chunked_source()" in patched[start:end]:
    raise RuntimeError("Preconverted setup wrapper did not replace the cold path")

# 09-01 诊断补丁: 伺服启动失败时原报错只带日志末120行, worker 的原始栈刚好被
# 切掉(合体版v2就死得无根因)。放大到4000行=整个启动日志, 失败必见根因。
_tail_old = "def tail(path: Path, lines: int = 120) -> str:"
if patched.count(_tail_old) != 1:
    raise RuntimeError("tail() anchor missing for diagnostic patch")
patched = patched.replace(_tail_old, "def tail(path: Path, lines: int = 4000) -> str:")

# 09-01 修复: expandable_segments 分配模式下, 词表卸载进程接收共享显存必须走
# pidfd_getfd 系统调用, 现被 Kaggle 沙箱禁止(合体v3全量日志实锤)。改回经典
# 分配模式, 共享走 CUDA IPC 句柄, 不碰该系统调用。代价=碎片化风险略升。
_alloc_old = '"PYTORCH_ALLOC_CONF": "expandable_segments:True",'
if patched.count(_alloc_old) != 1:
    raise RuntimeError("PYTORCH_ALLOC_CONF anchor missing for pidfd fix")
patched = patched.replace(_alloc_old, '"PYTORCH_ALLOC_CONF": "expandable_segments:False",')

# 09-01 修复3: Kaggle 新镜像系统 nvcc 过旧不认 SM120, flashinfer 现场编译算子时
# 报 "No supported CUDA architectures found for major versions [12]"(合体v4实锤)。
# 容器自带 CUDA 13.3 工具链(nvidia-cuda-nvcc==13.3.73), 把 CUDA_HOME 指过去;
# pip 布局可能缺 include/nvvm/lib64, 现场软链接补齐(布局已合并则全部跳过)。
_env_old = '"HF_HUB_OFFLINE": "1",\n        }\n    )\n    return env'
if patched.count(_env_old) != 1:
    raise RuntimeError("serving_env return anchor missing for CUDA_HOME fix")
_cuda_block = '''"HF_HUB_OFFLINE": "1",
        }
    )
    nvccs = sorted(SITE_PACKAGES.glob("nvidia/*/bin/nvcc")) + sorted(
        SITE_PACKAGES.glob("nvidia/*/*/bin/nvcc")
    )
    if nvccs:
        root = nvccs[0].parent.parent
        try:
            inc = root / "include"
            inc.mkdir(exist_ok=True)
            if not (inc / "cuda_runtime.h").exists():
                for src_inc in sorted(SITE_PACKAGES.glob("nvidia/*/include")):
                    for item in src_inc.iterdir():
                        dest = inc / item.name
                        if not dest.exists():
                            dest.symlink_to(item)
            if not (root / "nvvm").exists():
                for nvvm in sorted(SITE_PACKAGES.glob("nvidia/*/nvvm")):
                    (root / "nvvm").symlink_to(nvvm)
                    break
            lib64 = root / "lib64"
            if not lib64.exists():
                lib64.mkdir()
                for src_lib in sorted(SITE_PACKAGES.glob("nvidia/*/lib")):
                    for item in src_lib.iterdir():
                        dest = lib64 / item.name
                        if not dest.exists():
                            dest.symlink_to(item)
                for so in sorted(lib64.iterdir()):
                    name = so.name
                    while ".so." in name:
                        name = name.rsplit(".", 1)[0]
                        alias = lib64 / name
                        if not alias.exists():
                            alias.symlink_to(so)
        except OSError as exc:
            print(f"taaf.kaggle: bundled CUDA graft skipped: {exc}", flush=True)
        env["CUDA_HOME"] = str(root)
        env["PATH"] = str(root / "bin") + os.pathsep + env["PATH"]
        print(f"taaf.kaggle: CUDA_HOME -> {root}", flush=True)
    else:
        print("taaf.kaggle: no bundled nvcc found; CUDA_HOME unset", flush=True)
    return env'''
patched = patched.replace(_env_old, _cuda_block)




globals()["__file__"] = str(base_path)
exec(compile(patched, str(base_path), "exec"), globals(), globals())
