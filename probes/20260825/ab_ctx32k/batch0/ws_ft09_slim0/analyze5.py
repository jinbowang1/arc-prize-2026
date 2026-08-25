import json

with open('/tmp/grid.json') as f:
    g = json.load(f)
grid = g['grid']

# The color counts from the state don't match what I'm seeing. 
# State says: 色0 64, 色2 144, 色4 496, 色8 448, 色9 720, 色12 64
# But my analysis shows color 4 appears everywhere (columns 4-63 in rows 0-63).
# Something is wrong with how I'm reading the grid.

# Wait - the state summary says "连通块64个" (64 connected blocks)
# and color_account shows specific counts. Let me trust the state's color_account.
# Total: 64+144+496+448+720+64 = 1936 cells accounted for
# Grid has 64*64 = 4096 cells
# So there are unaccounted cells = 4096-1936 = 2160, which must be color 5 (background)

# So the actual colors are: 0, 2, 4, 5, 8, 9, 12
# Color 5 = background (2160 cells)
# Color 4 = 496 cells (not columns 4-63!)

# My earlier analysis was WRONG because I was looking at the raw grid values
# but those might not be what I think. Let me re-examine more carefully.

# Actually wait - the JSON grid values should be correct. Let me recount.
from collections import Counter
counts = Counter()
for r in range(64):
    for c in range(64):
        counts[grid[r][c]] += 1

print("Actual color counts from grid JSON:")
for color in sorted(counts.keys()):
    print(f"  Color {color}: {counts[color]}")

# These should match the state's color_account!
# If they don't, something weird is going on.
