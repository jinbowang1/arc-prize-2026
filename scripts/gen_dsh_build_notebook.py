"""生成 dsh 离线 bundle 构建 notebook (联网 CPU kernel).

产物 /kaggle/working/dsh-bundle.tgz = node22(linux-x64) + deepseek-harness
(fork 的空串分片修复分支, 含 node_modules 与构建产物)。提交 kernel 通过
kernel_sources 挂这个 kernel 的 output, 断网解包即用。
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

C1 = r'''import subprocess, os
from pathlib import Path
W = Path("/kaggle/working")
os.chdir(W)

def run(cmd, **kw):
    print("+", cmd, flush=True)
    subprocess.check_call(cmd, shell=True, **kw)

NODE = "node-v22.12.0-linux-x64"
run(f"wget -q https://nodejs.org/dist/v22.12.0/{NODE}.tar.xz")
run(f"tar xf {NODE}.tar.xz && rm {NODE}.tar.xz && mv {NODE} node")
os.environ["PATH"] = f"{W}/node/bin:" + os.environ["PATH"]
run("node --version")
run("npm install -g --force pnpm@10")  # 不走 corepack: shim 会与全局安装打架
run("pnpm --version")'''

C2 = r'''run("git clone --depth 1 -b fix/tool-call-empty-string-deltas "
    "https://github.com/jinbowang1/deepseek-harness dsh-src")
os.chdir(W / "dsh-src")
env = dict(os.environ, DSH_LEFTHOOK_ALLOW_HOOKS_PATH_OVERRIDE="1", CI="1")
subprocess.check_call("pnpm install --frozen-lockfile", shell=True, env=env)
subprocess.check_call("pnpm run build", shell=True, env=env)
print("build done", flush=True)'''

C3 = r'''os.chdir(W)
# 剔除 .git 和平台无关的大缓存; 保留 node_modules 符号链接结构(tar 原生支持)
run("rm -rf dsh-src/.git")
run("tar czf dsh-bundle.tgz node dsh-src")
run("rm -rf node dsh-src")
run("ls -lh dsh-bundle.tgz")
# 冒烟: 解包后 bin.js 能 --version (在本 kernel 内验证包完整)
run("mkdir -p /tmp/smoke && tar xzf dsh-bundle.tgz -C /tmp/smoke")
run("DSH_HOME=/tmp/smoke/home /tmp/smoke/node/bin/node "
    "/tmp/smoke/dsh-src/apps/cli/lib/bin.js --version")
print("bundle 冒烟通过", flush=True)'''


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
        cell("# dsh 离线 bundle 构建(联网 CPU kernel)\n\n"
             "产物 dsh-bundle.tgz 供提交 kernel 断网解包使用。", "markdown"),
        cell(C1), cell(C2), cell(C3),
    ],
}
out = ROOT / "kaggle_agent" / "notebook" / "arc3-dsh-build.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print("written:", out)
