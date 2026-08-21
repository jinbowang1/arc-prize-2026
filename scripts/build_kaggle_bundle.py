#!/usr/bin/env python3
"""打包 Kaggle 提交物: dataset bundle + kernel 目录。

用法:
  KAGGLE_USERNAME=<你的kaggle用户名> uv run python scripts/build_kaggle_bundle.py

产出(全部在 dist/ 下, git 忽略):
  dist/dataset/   -> kaggle datasets create -p dist/dataset   (首次)
                     kaggle datasets version -p dist/dataset -m "..."  (更新)
  dist/kernel/    -> kaggle kernels push -p dist/kernel
最后一步(提交)在 Kaggle 网页上: 打开 notebook -> Submit to Competition。
每天限交 1 次, 提交动作永远由人拍板, 脚本不做。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
USERNAME = os.environ.get("KAGGLE_USERNAME", "KAGGLE_USERNAME_未设置")
DATASET_SLUG = "arc3-jinbo-src"
KERNEL_SLUG = "arc3-jinbo-submission"
SAMPLE_GAMES = ["ft09", "ls20"]  # 离线冒烟兜底用的公开环境样本


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def build() -> None:
    ds = DIST / "dataset"
    kn = DIST / "kernel"
    for d in (ds, kn):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    # --- dataset: 源码 bundle ---
    shutil.copytree(
        ROOT / "kaggle_agent", ds / "kaggle_agent",
        ignore=shutil.ignore_patterns("__pycache__", "notebook"),
    )
    for g in SAMPLE_GAMES:
        src = ROOT / "environment_files" / g
        if src.is_dir():
            shutil.copytree(src, ds / "environment_files_sample" / g,
                            ignore=shutil.ignore_patterns(".DS_Store"))
    (ds / "arc3-jinbo-bundle.json").write_text(json.dumps({
        "name": "arc3-jinbo-src",
        "git_head": _git_head(),
    }, indent=1))
    (ds / "dataset-metadata.json").write_text(json.dumps({
        "title": DATASET_SLUG,
        "id": f"{USERNAME}/{DATASET_SLUG}",
        "licenses": [{"name": "CC-BY-4.0"}],
    }, indent=1))

    # --- kernel: notebook + 元数据 ---
    shutil.copy(ROOT / "kaggle_agent" / "notebook" / "arc3-jinbo-submission.ipynb", kn)
    (kn / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{USERNAME}/{KERNEL_SLUG}",
        "title": KERNEL_SLUG,
        "code_file": "arc3-jinbo-submission.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": [f"{USERNAME}/{DATASET_SLUG}"],
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
    }, indent=1))

    print(f"bundle ok (git {_git_head()}, user {USERNAME})")
    print("下一步:")
    print("  1. kaggle datasets create -p dist/dataset      # 首次; 更新用 datasets version")
    print("  2. kaggle kernels push -p dist/kernel")
    print("  3. 网页打开 notebook -> Save & Run 验证 -> Submit to Competition (人工拍板)")


if __name__ == "__main__":
    build()
