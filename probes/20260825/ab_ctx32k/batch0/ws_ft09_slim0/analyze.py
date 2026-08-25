import json

with open('/tmp/grid.json') as f:
    g = json.load(f)

# Let me save the grid first
grid = g['grid']

# Analyze the structure - look for patterns
# The 16x16 thumbnail shows interesting regions. Let me map out what's where.

# Print the full grid in a readable format, focusing on non-5 areas
print("=== Non-background regions ===")
for r in range(64):
    row_str = ""
    for c in range(64):
        row_str += str(grid[r][c])
    # Only print rows that have non-5 content
    if any(c != '5' for c in row_str):
        print(f"Row {r:2d}: {row_str}")

print("\n=== Color counts ===")
from collections import Counter
counts = Counter()
for r in range(64):
    for c in range(64):
        counts[grid[r][c]] += 1
for color in sorted(counts.keys()):
    print(f"Color {color}: {counts[color]}")
