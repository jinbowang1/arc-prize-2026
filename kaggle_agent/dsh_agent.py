"""dsh(deepseek-harness) 打一局: 进程内挂 GameServer, 起 dsh headless 子进程读任务书驱动。

与 run_submission 的其它 agent 同一契约: 吃 ApiGame, 吐 GameResult。
一局一个端口, 任务书里的端口号按局替换, 所以 N 局可以同时跑(赛场并发)。

配置全走环境变量(notebook 解包 dsh 后设好):
    A3_DSH_NODE   node 可执行文件
    A3_DSH_BIN    dsh 的 apps/cli/lib/bin.js
    A3_DSH_PATCH  vllm.patch.yml(模型层指向本地 vLLM)
    A3_DSH_TASK   任务书 TASK_FULL.md(默认取包内 kaggle_agent/dsh/TASK_FULL.md)
    A3_DSH_HOME   DSH_HOME 根(每局再分子目录, 会话账本互不串)
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from .explorer import GameResult
from .game_server import GameServer
from .remote_env import ApiGame

_PKG = Path(__file__).resolve().parent
_TASK_PORT_PLACEHOLDER = "18999"  # 任务书里写死的示例端口, 按局替换


def dsh_config() -> dict:
    cfg = {
        "node": os.environ.get("A3_DSH_NODE", ""),
        "bin": os.environ.get("A3_DSH_BIN", ""),
        "patch": os.environ.get("A3_DSH_PATCH", ""),
        "task": os.environ.get("A3_DSH_TASK", str(_PKG / "dsh" / "TASK_FULL.md")),
        "home": os.environ.get("A3_DSH_HOME", "/kaggle/tmp/dsh-home"),
    }
    missing = [k for k in ("node", "bin", "patch") if not cfg[k] or not Path(cfg[k]).exists()]
    if missing:
        raise RuntimeError(f"dsh 未配置或文件不存在: {missing} -> {cfg}")
    return cfg


def play_game_dsh(
    game: ApiGame,
    max_actions: int,
    deadline: float,
    port: int,
    out_dir: Path,
    cfg: dict | None = None,
) -> GameResult:
    cfg = cfg or dsh_config()
    gid = game.game_id
    short = gid.split("-")[0]
    t0 = time.monotonic()
    gs = GameServer(game, max_actions).start(port)
    task = Path(cfg["task"]).read_text().replace(_TASK_PORT_PLACEHOLDER, str(port))
    home = Path(cfg["home"]) / short
    ws = out_dir / "dsh_ws" / short
    home.mkdir(parents=True, exist_ok=True)
    ws.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, DSH_HOME=str(home), DEEPSEEK_API_KEY=os.environ.get("DEEPSEEK_API_KEY", "local"),
               # Kaggle 容器本身是一次性沙箱; dsh 的 Landlock/bubblewrap 在这不可用
               DSH_PERMISSION_MODE=os.environ.get("DSH_PERMISSION_MODE", "danger-full-access"))
    log = open(out_dir / f"dsh_{short}.log", "w")
    proc = subprocess.Popen(
        [cfg["node"], cfg["bin"], "--profile", "headless", "--patch", cfg["patch"], task],
        cwd=ws, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    state = "NOT_FINISHED"
    try:
        while proc.poll() is None:
            if time.monotonic() >= deadline or gs.obs.done or game.steps >= max_actions:
                break
            time.sleep(2)
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                proc.kill()
            state = "TIMEOUT"
        else:
            state = "AGENT_EXIT"
    finally:
        log.close()
        gs.stop()
    o = gs.obs
    return GameResult(
        game_id=gid,
        levels_completed=o.level,
        win_levels=o.win_levels,
        steps=game.steps,
        state="WIN" if o.done else state,
        per_level_steps=list(gs.per_level_steps),
        seconds=round(time.monotonic() - t0, 1),
    )
