"""运行级验收: 按关续命补丁跑在线上整包的真 _HarnessGameSession 上。

验五件:
  · 额度 = base + 关数 x bonus, 且**能超过原闸刀 7920**(这是这一发的全部意义)
  · 0 关的局只拿 base, 到点收工
  · 未到额度不收工
  · timing_payload 与真实额度一致(否则被续命的局会一直显示"还剩 0 秒")
  · 平均额度守恒: 按实测每局 2 关算, 不超过原闸刀
"""
import json, os, sys, types
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

# 线上 solver.py 依赖比赛的 arcengine 包(本地没有), 注入桩模块:
# 只需要 _HarnessGameSession 这个类本身, 补丁是往它上面挂方法。
_stub = types.ModuleType("inference.framework.solver")
class _HarnessGameSession:
    def runtime_limit_reached(self): return False
    def timing_payload(self): return {}
    def should_stop(self): return False
_stub._HarnessGameSession = _HarnessGameSession
for name in ("inference", "inference.framework"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["inference.framework.solver"] = _stub
solver = _stub

NB = ROOT / "kaggle_agent/duck/kernels/k_levelclock/arc3-levelclock.ipynb"
PATCH = "".join(json.loads(NB.read_text(encoding="utf-8"))["cells"][12]["source"])
PATCH = PATCH[PATCH.index("# ---- 09-17 按关续命"):]

R = []
def check(name, cond, extra=""):
    R.append((name, bool(cond))); print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  '+extra) if extra else ''}")

class _Solver:  max_runtime_s_per_game = 7920.0
class _BM:      solver = _Solver()
class _State:
    def __init__(self, lv): self.levels_completed = lv
class _Game:
    def __init__(self, lv, gid="tst-0000"):
        self.current_state = _State(lv)
        self.game_run = types.SimpleNamespace(game_id=gid)

ns = {"os": os, "bm": _BM()}
exec(compile(PATCH, "<lc>", "exec"), ns)
LC = ns["LEVELCLOCK"]

print("\n[1] 补丁生效")
check("已 patch", LC["patched"])
check("base=3000 bonus=2000", LC["base_s"] == 3000 and LC["bonus_s"] == 2000, f"{LC['base_s']}/{LC['bonus_s']}")
check("读到原闸刀 7920", LC["nominal"] == 7920.0)

print("\n[2] 额度随关数增长, 且能超过原闸刀")
def sess(lv, elapsed):
    s = solver._HarnessGameSession.__new__(solver._HarnessGameSession)
    s.solver = _Solver(); s.game = _Game(lv)
    s.started_at = ns["_lc_time"].monotonic() - elapsed
    return s
caps = {lv: ns["_lc_cap"](sess(lv, 0)) for lv in (0, 1, 2, 3, 5, 8)}
for lv, c in caps.items():
    print(f"        {lv} 关 -> 额度 {c:,.0f}s" + ("  ← 超过原闸刀" if c > 7920 else ""))
check("0 关只拿 base", caps[0] == 3000)
check("每关续 2000", caps[2] - caps[1] == 2000 and caps[3] - caps[2] == 2000)
check("🚨 3 关起超过原闸刀 7920(不封顶)", caps[3] > 7920 and caps[8] > 15000, f"3关{caps[3]:,.0f} 8关{caps[8]:,.0f}")
check("平均 2 关时额度 < 原闸刀(总墙钟不会超)", caps[2] < 7920, f"{caps[2]:,.0f} < 7920")

print("\n[3] 收工判据")
check("0 关跑到 2999s 不收工", not ns["_lc_runtime_limit_reached"](sess(0, 2999)))
check("0 关跑到 3001s 收工", ns["_lc_runtime_limit_reached"](sess(0, 3001)))
check("5 关跑到 7920s 不收工(原来会被砍)", not ns["_lc_runtime_limit_reached"](sess(5, 7920)))
check("5 关跑到 13001s 才收工", ns["_lc_runtime_limit_reached"](sess(5, 13001)))
check("收工记录里区分了'被续命'的局", any(r["cap"] > 7920 for r in LC["extended"]))

print("\n[4] 模型看到的剩余时间与真实额度一致")
for lv, el in ((0, 1000), (3, 5000), (5, 9000)):
    p = ns["_lc_timing_payload"](sess(lv, el))
    want = 3000 + lv*2000 - el
    ok = abs(p["time_remaining_seconds"] - want) < 2
    check(f"{lv} 关跑了 {el}s -> 还剩 {p['time_remaining_seconds']:,.0f}s", ok, f"应为 {want:,}")

print("\n[5] 边界")
s = sess(0, 100); s.solver = types.SimpleNamespace(max_runtime_s_per_game=None)
check("无限时配置下不误杀", not ns["_lc_runtime_limit_reached"](s))
check("无限时配置下 remaining=None", ns["_lc_timing_payload"](s)["time_remaining_seconds"] is None)

ok = sum(1 for _, c in R if c)
print(f"\n{ok}/{len(R)} 项通过")
sys.exit(0 if ok == len(R) else 1)
