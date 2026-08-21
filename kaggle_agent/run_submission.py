"""提交主控: 建 Arcade + 单 scorecard, 逐游戏跑 explorer, 汇总落盘。

双模式同一条代码路径:
- COMPETITION: KAGGLE_IS_COMPETITION_RERUN=1 且 ARC_BASE_URL 已设 ->
  网关拿隐藏游戏列表, 计分在网关侧 scorecard(中途盲, 只能开一张)。
- OFFLINE: 其余情况 -> 本地公开环境文件, 本地 scorecard, 可对账。

本地用法:
  uv run python -m kaggle_agent.run_submission --env-dir environment_files \
      --games ft09,ls20 --seconds-per-game 60 --max-actions 400
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# 🚨两个 env 必须在 import arc_agi 之前钉死:
# - ONLY_RESET_LEVELS: RESET=关卡重置(比赛规则), client 构建时缓存
# - OPERATION_MODE: 公司网墙下 OFFLINE 免去每次 make 等 arcprize SSL 超时
_IS_RERUN = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true"}
_COMPETITION = _IS_RERUN and bool(os.environ.get("ARC_BASE_URL"))
os.environ["ONLY_RESET_LEVELS"] = "true"
os.environ["OPERATION_MODE"] = "competition" if _COMPETITION else "offline"

import arc_agi  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402

from .explorer import play_game  # noqa: E402
from .remote_env import ApiGame  # noqa: E402


def _build_arcade(env_dir: str) -> arc_agi.Arcade:
    if _COMPETITION:
        return arc_agi.Arcade(
            arc_api_key=os.environ.get("ARC_API_KEY", ""),
            arc_base_url=os.environ["ARC_BASE_URL"],
            operation_mode=OperationMode.COMPETITION,
            environments_dir="",
        )
    return arc_agi.Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=env_dir)


def _prime_offline_scorecard(arcade, card_id: str, gid: str, env) -> None:
    """离线对账补丁(竞赛模式不需要, 也不执行)。

    实测坑: ONLY_RESET_LEVELS=true 让首次 RESET 的 full_reset=False ->
    Scorecard.new_play 永远不触发 -> 卡不建, take_action 全部漏记,
    离线 scorecard 恒为 0(空结果会骗人)。竞赛模式没这个问题: 服务端开
    竞赛 scorecard 时就预建全部环境(arc_agi/api.py "create all the
    environments for it")。这里只在离线模式手工建卡, 让本地对账成立。
    """
    if _COMPETITION:
        return
    guid = getattr(env, "_guid", None)
    card = arcade.scorecard_manager.scorecards.get(card_id)
    # ⚠️键必须用 wrapper 自己上报的 game_id(带版本号, 如 "ft09-0d8bbf25");
    # 用裸 gid 建卡, take_action 的 `if game_id in self.cards` 永远匹配不上
    full_gid = getattr(getattr(env, "environment_info", None), "game_id", gid)
    if guid and card is not None and full_gid not in card.cards:
        card.new_play(full_gid, guid)


def main(
    env_dir: str = "environment_files",
    games: list[str] | None = None,
    seconds_per_game: float = 300.0,
    max_actions: int = 1200,
    total_seconds: float | None = None,
    out_dir: str = ".",
) -> dict:
    t0 = time.monotonic()
    hard_deadline = t0 + total_seconds if total_seconds else None
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    arcade = _build_arcade(env_dir)
    infos = {i.game_id: i for i in arcade.get_environments()}
    game_ids = games or sorted(infos)
    if not game_ids:
        raise RuntimeError("环境列表为空(网关未就绪或 env_dir 不对)")
    print(f"mode={'COMPETITION' if _COMPETITION else 'OFFLINE'} games={len(game_ids)}")

    card_id = arcade.open_scorecard(tags=["jinbo-explorer-v1"])
    results = []
    for n, gid in enumerate(game_ids):
        # 剩余时间均分给剩余游戏, 不让前面的游戏吃光墙钟
        per = seconds_per_game
        if hard_deadline is not None:
            per = min(per, max(30.0, (hard_deadline - time.monotonic()) / (len(game_ids) - n)))
        env = arcade.make(gid, scorecard_id=card_id)
        if env is None:
            print(f"[{gid}] make 返回 None, 跳过")
            continue
        _prime_offline_scorecard(arcade, card_id, gid, env)
        # 单游戏隔离: 一个坏游戏(环境文件残缺/网关抽风)不许拖垮整场提交
        try:
            res = play_game(ApiGame(env, gid), max_actions=max_actions, deadline=time.monotonic() + per)
        except Exception as e:  # noqa: BLE001
            print(f"[{gid}] ERROR: {e!r}")
            results.append({"game_id": gid, "levels_completed": 0, "win_levels": 0,
                            "steps": 0, "state": f"ERROR: {e!r}", "per_level_steps": [], "seconds": 0.0})
            continue
        results.append(res.to_dict())
        print(
            f"[{gid}] levels {res.levels_completed}/{res.win_levels} "
            f"steps={res.steps} state={res.state} {res.seconds}s"
        )

    summary: dict = {
        "mode": "competition" if _COMPETITION else "offline",
        "card_id": card_id,
        "games": results,
        "total_levels": sum(r["levels_completed"] for r in results),
        "wall_seconds": round(time.monotonic() - t0, 1),
    }
    # 竞赛 scorecard 中途盲: close/get 拿不到明细也不能让整次提交垮掉
    try:
        card = arcade.close_scorecard(card_id)
        if card is not None:
            summary["scorecard"] = json.loads(card.model_dump_json())
    except Exception as e:  # noqa: BLE001
        summary["scorecard_error"] = repr(e)

    (out / "results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"total levels={summary['total_levels']} wall={summary['wall_seconds']}s -> {out/'results.json'}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-dir", default="environment_files")
    ap.add_argument("--games", default="")
    ap.add_argument("--seconds-per-game", type=float, default=300.0)
    ap.add_argument("--max-actions", type=int, default=1200)
    ap.add_argument("--total-seconds", type=float, default=None)
    ap.add_argument("--out-dir", default=".")
    a = ap.parse_args()
    main(
        env_dir=a.env_dir,
        games=[g for g in a.games.split(",") if g] or None,
        seconds_per_game=a.seconds_per_game,
        max_actions=a.max_actions,
        total_seconds=a.total_seconds,
        out_dir=a.out_dir,
    )
