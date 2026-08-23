"""生成行动密度 A/B notebook (覆写 arc3-jinbo-llm-smoke.ipynb).

A/B: SYSTEM_PROMPT old vs +"行动要成段"指引, Qwen 本尊同机串行,
r11l+ls20 各 400s/30轮/100动作。判据: 过关数 > steps(行动密度) > act 密度。
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
assert torch.cuda.get_device_capability(0) >= (8, 9), "FP8 需要 CC>=8.9 (T4/P100 不行)"'''

C2 = '''def find_wheelhouse(pattern):
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
assert model_dir, "找不到模型目录(要挂 Kaggle Model)"
bundle = next(Path("/kaggle/input").rglob("arc3-jinbo-bundle.json")).parent
sys.path.insert(0, str(bundle))
print("model:", model_dir, "| bundle:", bundle)'''

C3 = '''from kaggle_agent.serve_vllm import start_vllm
proc = start_vllm(str(model_dir), port=8000, max_model_len=32768,
                  log_path=str(WORKING / "vllm.log"))
from kaggle_agent.llm import LLMClient
smoke = LLMClient("http://127.0.0.1:8000", model="local", max_tokens=100)
print("冒烟:", smoke.chat([{"role": "user", "content": "回复OK"}])[:50])

extra = None
try:
    nt = LLMClient("http://127.0.0.1:8000", model="local", max_tokens=2000,
                   extra={"chat_template_kwargs": {"enable_thinking": False}})
    if nt.chat([{"role": "user", "content": "9+13*7等于几? 只回答数字"}]).strip():
        extra = {"chat_template_kwargs": {"enable_thinking": False}}
        print("已启用关思考模式(与提交版一致)")
except Exception as e:
    print("关思考探测失败, 带思考跑:", repr(e))'''

# ⚠️EXTRA_RULE 文案与 probes/20260823/ab_actiondensity.py 保持逐字一致
C4 = '''from kaggle_agent.run_submission import _build_arcade
from kaggle_agent.remote_env import ApiGame
from kaggle_agent import repl_agent

EXTRA_RULE = """6. 行动要成段: 方向明确后, 写带循环的代码让 act 连续执行到位(每步检查返回值
   和 grid, 与预期不符立即 break 回来分析), 不要一轮只走一两步 —— 你的开口
   次数比动作预算稀缺得多, 大多数游戏输在"来不及行动"而不是"动作用超"。"""
ANCHOR = "你在代码里定义的函数和变量会跨轮保留"
OLD = repl_agent.SYSTEM_PROMPT
NEW = OLD.replace(ANCHOR, EXTRA_RULE + "\\n" + ANCHOR)
assert NEW != OLD, "锚点没找到, A/B 无效"

env_dir = str(Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files"))
arcade = _build_arcade(env_dir)
gids = sorted(i.game_id for i in arcade.get_environments())
results = []
for arm, prompt in (("old", OLD), ("new", NEW)):
    repl_agent.SYSTEM_PROMPT = prompt
    for base in ("r11l", "ls20"):
        gid = next(x for x in gids if x.startswith(base))
        g = ApiGame(arcade.make(gid), gid)
        llm = LLMClient("http://127.0.0.1:8000", model="local", extra=extra)
        tp = WORKING / f"ab_{base}_{arm}.jsonl"
        res = repl_agent.play_game_repl(g, llm, max_actions=100,
                                        deadline=time.monotonic() + 400,
                                        max_rounds=30, transcript_path=str(tp))
        rows = [json.loads(l) for l in open(tp)]
        acts = [r.get("assistant", "").count("act(") for r in rows if "round" in r]
        rec = dict(arm=arm, game=base, levels=res.levels_completed, steps=g.steps,
                   rounds=len(acts), act_calls=sum(acts), llm_calls=llm.stats.calls,
                   seconds=res.seconds)
        print(rec)
        results.append(rec)
(WORKING / "ab_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
print("A/B done")'''


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
        cell("# ARC-AGI-3 行动密度 A/B — SYSTEM_PROMPT old vs +行动成段\n\n"
             "Qwen3.8-27B 本尊, 同机串行 r11l+ls20 × 两臂, 各 400s/30轮/100动作。\n"
             "开发用 notebook, 不是提交物。", "markdown"),
        cell(C1), cell(C2), cell(C3), cell(C4),
    ],
}
out = ROOT / "kaggle_agent" / "notebook" / "arc3-jinbo-llm-smoke.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print("written:", out)
