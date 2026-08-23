"""A/B: 行动成段指引 (old vs new SYSTEM_PROMPT), DeepSeek 陪练.

用法: uv run python probes/20260823/ab_actiondensity.py <game> <old|new>
判据: 过关数优先; 平级看 steps(行动密度)与每轮 act 数。
"""
import json, sys, time
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ))

from kaggle_agent.run_submission import _build_arcade  # noqa: E402
from kaggle_agent.remote_env import ApiGame  # noqa: E402
from kaggle_agent import repl_agent  # noqa: E402
from kaggle_agent.llm import LLMClient  # noqa: E402

EXTRA_RULE = """
6. 行动要成段: 方向明确后, 写带循环的代码让 act 连续执行到位(每步检查返回值
   和 grid, 与预期不符立即 break 回来分析), 不要一轮只走一两步 —— 你的开口
   次数比动作预算稀缺得多, 大多数游戏输在"来不及行动"而不是"动作用超"。
"""
ANCHOR = "你在代码里定义的函数和变量会跨轮保留"

game_name, arm = sys.argv[1], sys.argv[2]
assert arm in ("old", "new")
if arm == "new":
    assert ANCHOR in repl_agent.SYSTEM_PROMPT
    repl_agent.SYSTEM_PROMPT = repl_agent.SYSTEM_PROMPT.replace(
        ANCHOR, EXTRA_RULE.strip() + "\n" + ANCHOR)

MASTER_KEY = "sk-master-291a713ed023c50534b4eebbfa1accb3"  # 本机 LiteLLM, 不出网
llm = LLMClient("http://127.0.0.1:4000", model="aliyun/deepseek-v4-flash",
                api_key=MASTER_KEY)

arcade = _build_arcade(str(PROJ / "environment_files"))
gid = next(g for g in sorted(i.game_id for i in arcade.get_environments())
           if g.startswith(game_name))
env = arcade.make(gid)
g = ApiGame(env, gid)

out = Path(__file__).parent / f"ab_{game_name}_{arm}.jsonl"
out.unlink(missing_ok=True)
t0 = time.monotonic()
res = repl_agent.play_game_repl(
    g, llm, max_actions=150, deadline=time.monotonic() + 420,
    max_rounds=8, transcript_path=str(out))
rows = [json.loads(l) for l in open(out)]
acts_per_round = [r.get("assistant", "").count("act(") for r in rows if "round" in r]
print(json.dumps({
    "game": game_name, "arm": arm,
    "levels": res.levels_completed, "steps": g.steps,
    "rounds": len(acts_per_round), "act_calls_in_code": acts_per_round,
    "llm_calls": llm.stats.calls, "seconds": round(time.monotonic() - t0, 1),
}, ensure_ascii=False))
