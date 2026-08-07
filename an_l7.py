"""分析 l7_model.json: 可达格、真补给、致死格、未探边界。"""
import ast, json
from collections import defaultdict

d = json.load(open("l7_model.json"))
T = {ast.literal_eval(k): v for k, v in d["transitions"].items()}   # 值是 [状态串, dE, 标志]
START = (15, 19)          # 起点 = 死亡重生点

cells, moves, deaths, pickups = set(), {}, set(), defaultdict(int)
shapes, colors = set(), set()
for (state, a), v in T.items():
    cell, (sh, col) = state[0], state[1]
    cells.add(cell); shapes.add(sh); colors.add(col)
    nxt = ast.literal_eval(v[0])
    dst, dE = nxt[0], v[1]
    if dst == START and dE > 0 and cell != START:
        deaths.add((cell, a))          # 传送回起点 + 回能 = 死亡
        continue
    moves[(cell, a)] = dst
    cells.add(dst)
    if dE > 0 and dst != cell:
        pickups[dst] = max(pickups[dst], dE)

rows = sorted({r for r, _ in cells}); cols = sorted({c for _, c in cells})
print(f"形状集合 {shapes} | 颜色集合 {colors}")
print(f"可达格 {len(cells)} 个, 行 {rows}, 列 {cols}")
print(f"真补给 {dict(pickups)}")
print(f"致死 (格,动作) {len(deaths)} 条, 涉及 {len({c for c,_ in deaths})} 个格")

print("\n地图(o=可达 X=有致死出口 P=补给 S=起点 .=没到过):")
print("     " + " ".join(f"{c:>2}" for c in cols))
for r in rows:
    line = []
    for c in cols:
        cell = (r, c)
        if cell == START: line.append(" S")
        elif cell in pickups: line.append(" P")
        elif cell in {x for x, _ in deaths}: line.append(" X")
        elif cell in cells: line.append(" o")
        else: line.append("  .")
    print(f"{r:>4} " + " ".join(line))

print("\n每个格的出边(缺边=没试过或撞墙):")
miss = [(cell, a) for cell in sorted(cells) for a in (1, 2, 3, 4)
        if (cell, a) not in moves and (cell, a) not in deaths]
print(f"  缺 {len(miss)} 条, 例: {miss[:15]}")
