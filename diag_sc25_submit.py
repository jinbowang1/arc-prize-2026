"""L3: 那 5 个"只在画布涂过后激活"的按钮, 会不会是**提交/确认键**?

若是, 就能解释之前所有失败: 我枚举过 42 个方块位置 x 五种画布图案,
**从来没按过提交**。
按钮: A6(11,55) A6(12,55) A6(15,51) A6(15,54) A6(15,57)
"""
import json
import numpy as np
from harness.env import Action, Game
from harness.run import _parse

sol = json.load(open("sc25_solutions.json"))
game, obs = Game.make("sc25")
for t in sol["seq"]:
    obs = game.act(_parse(t))
L0 = obs.level
R, C = [49, 54, 59], [24, 29, 34]
cv = lambda g: [[int(np.array(g)[R[i]+1, C[j]+1]) for j in range(3)] for i in range(3)]
click = lambda i, j: Action.click(C[j] + 1, R[i] + 1)
BTNS = [Action.click(11, 55), Action.click(12, 55), Action.click(15, 51),
        Action.click(15, 54), Action.click(15, 57)]

print(f"起点 level={L0} 画布 {cv(obs.grid)}\n")
print("先涂画布, 再按各按钮, 看会不会过关:")
# 几种画布配置 x 5 个按钮
CONFIGS = {
    "中列点1次": [(i, 1) for i in range(3)],
    "中列点2次": [(i, 1) for i in range(3)] * 2,
    "边列点1次": [(i, j) for i in range(3) for j in (0, 2)],
    "全格点1次": [(i, j) for i in range(3) for j in range(3)],
    "不涂":      [],
}
for cname, cells in CONFIGS.items():
    for bi, btn in enumerate(BTNS):
        n = game.fork()
        ok = True
        for (i, j) in cells:
            if n.act(click(i, j)).dead:
                ok = False; break
        if not ok:
            continue
        o = n.act(btn)
        if o.dead:
            continue
        won = o.level > L0
        # 补一步查滞后
        p = n.fork().act(btn)
        won2 = (not p.dead) and p.level > L0
        if won or won2:
            print(f"  🏆 {cname} + 按钮{bi} {repr(btn)} -> **过关**"
                  f"{'(补步)' if won2 and not won else ''}")
        elif bi == 0:
            print(f"  {cname:<12} + 按钮0 -> level={o.level} 画布 {cv(o.grid)}")
