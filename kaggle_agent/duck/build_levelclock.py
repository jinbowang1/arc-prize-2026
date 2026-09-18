"""09-17 按关续命 (arc3-levelclock): 时间额度随过关增长, 上不封顶。

## 病根 (09-01 整跑 25 局实测)

**25 局在"最后一次过关之后"还烧掉 29.5 小时 = 总时间的 54%。**
而每局的闸刀是死的 7920 秒 —— 卡死的局和还在出关的局拿一样多的时间。

过关是有节奏的: 首关耗时中位 1,383s (25% 分位 457s / 75% 分位 2,607s / 90% 分位 3,166s);
后续每关间隔中位 1,505s。**0 关的局每个白烧 7,922s。**

🚨 首关慢 ≠ 这局不行: 首关 >2600s 的 7 局最终关数中位 2, 与首关 ≤1400s 的 12 局**相同**。
所以判据不能是"开局慢就砍", 必须是"不再进步才砍"。

## 改法

每局的时间额度 = `base_s + 已过关数 x bonus_s`, **不设上限**(原来死卡 7920)。
- 一关没过 -> 额度只有 base_s, 到点收工, 时间让给别人
- 每过一关 -> 续一笔, 还在出关的局可以跑到远超 7920

平均额度守恒: 实测平均每局 2 关 -> 3000 + 2x2000 = 7000 < 7920, 总墙钟只会缩短不会超。
而过 5 关的局能拿到 13,000 秒(原闸刀的 1.6 倍)。

🚨 **为什么必须上隐藏集测**: 公开集 25 局 < 28 并发, 一批跑完, 早停腾出的时间没有下一批可给,
反而因分母变小把每局时限放大到 11880s —— 09-14 的 k_bud 就是这么测出 43 关的, 那个数不作数。

同时改 timing_payload: 模型看到的"还剩多少秒"必须与真实额度一致。

只改 cell 12, 线上数据集一个字不动。用法: python3 build_levelclock.py -> kernels/k_levelclock
"""
from __future__ import annotations
import copy, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_NB = HERE / "kernels/k_fnrep_run/arc3-fnrep-run.ipynb"
BASE_META = HERE / "kernels/k_fnrep_run/kernel-metadata.json"
SLUG = "arc3-levelclock"

PATCH = r'''

# ---- 09-17 按关续命: 时间额度随过关增长, 上不封顶 ----
# 54% 的时间烧在"最后一次过关之后"。死闸刀让卡住的局和还在出关的局拿一样多的时间。
import time as _lc_time

import inference.framework.solver as _lc_solver

LEVELCLOCK = {
    "base_s":  float(os.environ.get("ARC3_LC_BASE", "3000")),    # 首关额度(覆盖约 88% 的首关)
    "bonus_s": float(os.environ.get("ARC3_LC_BONUS", "2000")),   # 每过一关续多少(后续每关间隔中位 1505)
    "nominal": float(getattr(bm.solver, "max_runtime_s_per_game", 0.0) or 7920.0),
    "stops": [], "extended": [], "patched": False,
}

_lc_orig_limit = _lc_solver._HarnessGameSession.runtime_limit_reached
_lc_orig_timing = _lc_solver._HarnessGameSession.timing_payload


def _lc_levels(self):
    try:
        return int(self.game.current_state.levels_completed)
    except Exception:
        return 0


def _lc_cap(self):
    """额度 = base + 已过关数 x bonus, 不封顶。"""
    return LEVELCLOCK["base_s"] + _lc_levels(self) * LEVELCLOCK["bonus_s"]


def _lc_runtime_limit_reached(self) -> bool:
    if self.solver.max_runtime_s_per_game is None:
        return False
    elapsed = _lc_time.monotonic() - self.started_at
    cap = _lc_cap(self)
    if elapsed < cap:
        return False
    lv = _lc_levels(self)
    run = getattr(self.game, "game_run", None)
    gid = getattr(run, "game_id", "?")
    rec = {"game": gid, "levels": lv, "elapsed": round(elapsed), "cap": round(cap)}
    LEVELCLOCK["stops"].append(rec)
    if cap > LEVELCLOCK["nominal"]:
        LEVELCLOCK["extended"].append(rec)
    print(f"LEVELCLOCK: 收工 game={gid} 已过 {lv} 关, 用了 {elapsed:.0f}s "
          f"(额度 {cap:.0f}s = {LEVELCLOCK['base_s']:.0f}+{lv}x{LEVELCLOCK['bonus_s']:.0f}"
          f"{', 超出原闸刀 ' + str(round(cap - LEVELCLOCK['nominal'])) + 's' if cap > LEVELCLOCK['nominal'] else ''})",
          flush=True)
    return True


def _lc_timing_payload(self):
    # 模型看到的"还剩多少秒"必须与真实额度一致, 否则被续命的局会一直显示 0 秒
    elapsed = max(0.0, _lc_time.monotonic() - self.started_at)
    if self.solver.max_runtime_s_per_game is None:
        return {"run_elapsed_seconds": elapsed, "time_remaining_seconds": None}
    return {"run_elapsed_seconds": elapsed,
            "time_remaining_seconds": max(0.0, _lc_cap(self) - elapsed)}


_lc_solver._HarnessGameSession.runtime_limit_reached = _lc_runtime_limit_reached
_lc_solver._HarnessGameSession.timing_payload = _lc_timing_payload
LEVELCLOCK["patched"] = True
print(f"LEVELCLOCK patched: 额度 = {LEVELCLOCK['base_s']:.0f} + 关数 x {LEVELCLOCK['bonus_s']:.0f}, "
      f"不封顶(原闸刀 {LEVELCLOCK['nominal']:.0f}s)", flush=True)
'''


def _set_source(cell, text):
    cell["source"] = text.splitlines(keepends=True)


def build(slug=SLUG, base=3000, bonus=2000):
    nb = json.loads(BASE_NB.read_text(encoding="utf-8"))
    base_nb = json.loads(BASE_NB.read_text(encoding="utf-8"))
    meta = json.loads(BASE_META.read_text(encoding="utf-8"))
    patch = PATCH.replace('"ARC3_LC_BASE", "3000"', f'"ARC3_LC_BASE", "{base}"') \
                 .replace('"ARC3_LC_BONUS", "2000"', f'"ARC3_LC_BONUS", "{bonus}"')
    cell12 = "".join(nb["cells"][12]["source"])
    assert "V31 concurrency" in cell12, "cell 12 锚点"
    _set_source(nb["cells"][12], cell12.rstrip("\n") + "\n" + patch)
    for i, (new, old) in enumerate(zip(nb["cells"], base_nb["cells"])):
        if i == 12: continue
        assert "".join(new["source"]) == "".join(old["source"]), f"cell {i} 不该动"
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []; cell["execution_count"] = None
    out = HERE / f"kernels/k_{slug.replace('arc3-','')}"
    out.mkdir(parents=True, exist_ok=True)
    m = copy.deepcopy(meta); m["id"] = f"jinbowang1/{slug}"; m.pop("id_no", None)
    m["title"] = slug; m["code_file"] = f"{slug}.ipynb"
    (out/"kernel-metadata.json").write_text(json.dumps(m, indent=2, ensure_ascii=False)+"\n")
    (out/f"{slug}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False)+"\n")
    print(f"built {out/f'{slug}.ipynb'}  (base={base} bonus={bonus})")
    return out


if __name__ == "__main__":
    build()
