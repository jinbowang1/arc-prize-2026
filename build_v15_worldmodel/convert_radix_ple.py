"""Convert the RadixArk Flash-Next PLE table from FP8 storage to BF16.

The experimental vLLM PLE CPU-offload worker expects ordinary BF16 embedding
weights.  RadixArk stores the same table as FP8 tensors plus one scalar scale.
This conversion leaves the NVFP4 routed experts and every other tensor exactly
as published; only the 128 PLE lookup shards are dequantized.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


PLE_PREFIX = (
    "model.language_model.layers.1.ple.ple_embedding."
    "ngram_embedding.shard_"
)
SCALE_KEY = (
    "model.language_model.layers.1.ple.ple_embedding."
    "ngram_embedding.weight_scale"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect(path: Path) -> tuple[set[str], dict[str, str] | None]:
    with safe_open(path, framework="pt", device="cpu") as fh:
        return set(fh.keys()), fh.metadata()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--delete-fp8", action="store_true")
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    index_path = model_dir / "model.safetensors.index.json"
    manifest_path = model_dir / "ple-bf16-conversion.json"
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("status") == "complete":
            print("PLE BF16 conversion already complete", flush=True)
            return

    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map: dict[str, str] = index["weight_map"]
    ple_keys = sorted(k for k in weight_map if k.startswith(PLE_PREFIX))
    if len(ple_keys) != 128:
        raise RuntimeError(f"expected 128 PLE shard tensors, found {len(ple_keys)}")
    if SCALE_KEY not in weight_map:
        raise RuntimeError(f"missing {SCALE_KEY}")

    scale_file = model_dir / weight_map[SCALE_KEY]
    scale_tensor = load_file(scale_file, device="cpu")[SCALE_KEY]
    scale = float(scale_tensor.float().item())
    del scale_tensor
    gc.collect()
    if not 0.0 < scale < 1.0:
        raise RuntimeError(f"implausible PLE scale {scale}")
    print(f"PLE scale: {scale:.12g}", flush=True)

    source_names = sorted({weight_map[k] for k in ple_keys})
    if len(source_names) != 10:
        raise RuntimeError(f"expected 10 PLE files, found {len(source_names)}")
    if scale_file.name not in source_names:
        raise RuntimeError("PLE scale is not colocated with the published PLE shards")

    free = shutil.disk_usage(model_dir).free
    source_bytes = sum((model_dir / name).stat().st_size for name in source_names)
    # BF16 is two bytes/value while the published PLE table is one byte/value.
    # Keep both copies until every output shard and the rewritten index verify.
    expected_extra = 2 * source_bytes
    if free < expected_extra + 12 * 1024**3:
        raise RuntimeError(
            f"insufficient free disk: {free / 1024**3:.1f} GiB available, "
            f"need at least {(expected_extra + 12 * 1024**3) / 1024**3:.1f} GiB"
        )

    converted_files: list[dict[str, object]] = []
    new_map = dict(weight_map)
    for ordinal, source_name in enumerate(source_names):
        source = model_dir / source_name
        target_name = source_name.replace("model-plefp8-", "model-plebf16-")
        target = model_dir / target_name
        temporary = target.with_suffix(target.suffix + ".tmp")

        source_keys, metadata = inspect(source)
        expected_keys = {k for k in ple_keys if weight_map[k] == source_name}
        file_weight_keys = {k for k in source_keys if k.startswith(PLE_PREFIX)}
        if file_weight_keys != expected_keys:
            raise RuntimeError(
                f"index/header mismatch for {source_name}: "
                f"index={len(expected_keys)} header={len(file_weight_keys)}"
            )

        if target.exists():
            target_keys, _ = inspect(target)
            if target_keys != expected_keys:
                raise RuntimeError(f"stale or incomplete target {target}")
            print(f"[{ordinal + 1}/10] reusing {target.name}", flush=True)
        else:
            print(f"[{ordinal + 1}/10] converting {source.name}", flush=True)
            tensors = load_file(source, device="cpu")
            converted: dict[str, torch.Tensor] = {}
            for key in sorted(expected_keys):
                tensor = tensors[key]
                if not str(tensor.dtype).startswith("torch.float8"):
                    raise RuntimeError(f"unexpected dtype for {key}: {tensor.dtype}")
                converted[key] = tensor.to(torch.bfloat16).mul_(scale).contiguous()
            save_file(converted, temporary, metadata=metadata)
            os.replace(temporary, target)
            target_keys, _ = inspect(target)
            if target_keys != expected_keys:
                raise RuntimeError(f"verification failed for {target}")
            del tensors, converted
            gc.collect()

        for key in expected_keys:
            new_map[key] = target_name
        converted_files.append(
            {
                "source": source_name,
                "target": target_name,
                "target_bytes": target.stat().st_size,
                "target_sha256": sha256(target),
                "tensor_count": len(expected_keys),
            }
        )

    del new_map[SCALE_KEY]
    index["weight_map"] = new_map
    if isinstance(index.get("metadata"), dict) and "total_size" in index["metadata"]:
        target_bytes = sum(int(item["target_bytes"]) for item in converted_files)
        old_total = int(index["metadata"]["total_size"])
        index["metadata"]["total_size"] = old_total - source_bytes + target_bytes
    temporary_index = index_path.with_suffix(index_path.suffix + ".tmp")
    temporary_index.write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_index, index_path)

    if args.delete_fp8:
        for source_name in source_names:
            (model_dir / source_name).unlink()

    manifest = {
        "status": "complete",
        "conversion": "FP8 E4M3 values multiplied by checkpoint scalar to BF16",
        "scale": scale,
        "source_tensor_count": len(ple_keys),
        "scale_key_removed": SCALE_KEY,
        "fp8_sources_deleted": bool(args.delete_fp8),
        "files": converted_files,
        "index_sha256": sha256(index_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
