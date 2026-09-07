"""把 kernel 公开集输出对着第十一发(5.49)的基线判, 并折算隐藏集预期。

基线(第十一发 55955743 公开集): 均分 9.11 / 48 关(共174) / 23 局开张 / 4902 步 -> 隐藏 5.49, 折算比 0.60
"""
import json, sys, os, glob

BASE = dict(score=9.11, levels=48, total_levels=174, opened=23, games=25, actions=4902,
            hidden=5.49, ratio=0.60)

def load(path):
    """从 kernel 输出目录里找 benchmark.json / summary。"""
    cands = []
    if os.path.isdir(path):
        cands = glob.glob(os.path.join(path, "**", "benchmark.json"), recursive=True)
    elif path.endswith(".json"):
        cands = [path]
    for c in cands:
        try:
            d = json.load(open(c, encoding="utf-8"))
        except Exception:
            continue
        runs = d.get("runs") or d.get("results") or []
        if runs:
            return d, runs, c
    return None, None, None

def main(path):
    d, runs, src = load(path)
    if not runs:
        print(f"没在 {path} 找到 benchmark.json 的 runs"); return
    n = len(runs)
    lv = sum(int(r.get("levels_completed", r.get("levels", 0)) or 0) for r in runs)
    sc = sum(float(r.get("score") or 0) for r in runs) / n
    ac = sum(int(r.get("actions", r.get("actions_taken", 0)) or 0) for r in runs)
    op = sum(1 for r in runs if (r.get("levels_completed", r.get("levels", 0)) or 0) > 0)
    print(f"来源: {src}")
    print(f"\n{'':10}{'winframe':>12}{'基线(5.49)':>14}{'差':>10}")
    print("-" * 48)
    print(f"{'局数':10}{n:>12}{BASE['games']:>14}{n-BASE['games']:>+10}")
    print(f"{'过关数':9}{lv:>12}{BASE['levels']:>14}{lv-BASE['levels']:>+10}")
    print(f"{'开张局':9}{op:>12}{BASE['opened']:>14}{op-BASE['opened']:>+10}")
    print(f"{'均分':10}{sc:>12.2f}{BASE['score']:>14.2f}{sc-BASE['score']:>+10.2f}")
    print(f"{'总步数':9}{ac:>12}{BASE['actions']:>14}{ac-BASE['actions']:>+10}")
    print(f"\n按折算比 {BASE['ratio']} 估隐藏集: {sc*BASE['ratio']:.2f}  (第十一发实际 {BASE['hidden']})")
    print("\n验收(计划里定的): 关数涨 且 开张不掉 -> " +
          ("✅ 通过" if lv >= BASE['levels'] and op >= BASE['opened'] else "❌ 未过"))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
