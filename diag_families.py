"""ls20 卡在 h=3, 而 region_match 表达不了它的真判据(钥匙形状+颜色匹配锁)。
harness 已有 ObjectToObject 族("把形状 A 的对象移到形状 B 处") —— 它被提出来了吗?
没有的话, 是哪一步把它挡掉了?"""
import numpy as np
from harness import hypo
from harness.env import Action, Game, action_space
from harness.percept import analyze, discover
from harness.probe import run_probe

for gid in ("ls20", "cd82"):
    game, obs = Game.make(gid)
    sp = action_space(list(obs.actions))
    sc = analyze(obs.grid)
    clicks = [Action.click(c, r) for (r, c) in sc.targets]
    acts = [Action.key(i) for i in sp["keys"]] + clicks
    game.detect_lag(acts)
    rep = run_probe(game, obs, sp["kind"], sp["keys"], clicks)
    ents, _ = discover(lambda a: np.array(game.effect(a).grid), np.array(obs.grid), acts)
    print(f"\n=== {gid} === 实体 {len(ents)} 个")
    for e in ents[:6]:
        print(f"   {e}")
    # 各族分别能提出多少条
    for name, fam in hypo.FAMILIES.items():
        try:
            props = fam.propose(np.array(obs.grid), ents, rep.mask) \
                if hasattr(fam, "propose") else []
            print(f"   {name:18} 提议 {len(props)} 条 {[str(p)[:50] for p in props[:2]]}")
        except Exception as e:
            print(f"   {name:18} 提议失败: {type(e).__name__}: {str(e)[:40]}")
