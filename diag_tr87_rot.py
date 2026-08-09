"""L1 规则精确化: 题面[i] = 变换T(src_j) 时, 答案[i] = 变换?(dst_j)。枚举全部匹配。"""
import numpy as np
from parse_tr87 import show

OPS = {"r0": lambda x: x, "r90": lambda x: np.rot90(x, -1), "r180": lambda x: np.rot90(x, 2),
       "r270": lambda x: np.rot90(x, 1), "fx": np.fliplr, "fy": np.flipud,
       "ft": lambda x: x.T, "fa": lambda x: np.rot90(x.T, 2)}

def M(s):
    return np.array([[1 if ch == "X" else 0 for ch in row] for row in s.split("/")])

DICT = [
    ("XXXXX/X...X/XX.XX/X...X/X...X", "XXXX./X..XX/X...X/XX..X/.XXXX"),
    ("X...X/XXXXX/..X../XXXXX/X...X", "XXXX./X..X./X..XX/X..X./XXXX."),
    ("....X/..X.X/XXXXX/..X.X/....X", "XXXXX/X..X./X..X./XXXX./X...."),
    ("..X../XXXXX/X.X.X/X.X.X/..X..", "..XXX/..X.X/XXXXX/X.X../XXX.."),
    ("....X/..X.X/XXXXX/X.X../X....", "XXXXX/X...X/X.XXX/X.X.X/XXXXX"),
    ("XX.XX/X...X/X...X/XXXXX/X...X", "..X../XXXXX/X.X.X/XXXXX/..X.."),
]
PROB = ["....X/.XXXX/.X..X/.X..X/XXXXX",  # 这里放 L1 题面
        "....X/..X.X/XXXXX/X.X../X....",
        "XXXXX/..X.X/....X/..X.X/XXXXX",
        "XX.XX/.X.X./.XXX./.X.X./XX.XX",
        "..X../..X../.XXX./..X../XXXXX"]
PROB[0] = ".XXX./...X./XXXXX/...X./.XXX."
ANS = ["..XXX/..X.X/XXXXX/X.X../XXX..",
       "XXXXX/X.X.X/XXX.X/X...X/XXXXX",
       "XXXX./X..XX/X...X/XX..X/.XXXX",
       "..X../XXXXX/X...X/X...X/XXXXX",
       "XXXXX/.X..X/.X..X/.XXXX/....X"]

for i, (p, a) in enumerate(zip(PROB, ANS)):
    pm, am = M(p), M(a)
    pmatch = [(j + 1, op) for j, (s, d) in enumerate(DICT) for op, f in OPS.items()
              if np.array_equal(f(M(s)), pm)]
    amatch = [(j + 1, op) for j, (s, d) in enumerate(DICT) for op, f in OPS.items()
              if np.array_equal(f(M(d)), am)]
    print(f"位{i+1}: 题面匹配 {pmatch}  | 答案匹配 {amatch}")
