import json

with open('/tmp/grid.json') as f:
    g = json.load(f)
grid = g['grid']

# The 16x16 downsampled view doesn't match the state summary at all.
# Let me re-examine. The state summary shows a different 16x16 view.
# Let me check what colors are actually in each 4x4 block by looking at the majority color.

print("=== Majority-color 16x16 view ===")
thumb = []
for r in range(16):
    row = []
    for c in range(16):
        counts = {}
        for dr in range(4):
            for dc in range(4):
                val = grid[r*4+dr][c*4+dc]
                counts[val] = counts.get(val, 0) + 1
        # Pick the most common color
        major = max(counts.items(), key=lambda x: x[1])
        row.append(str(major[0]))
    thumb.append(''.join(row))
    print(f"Row {r}: {' '.join(row)}")

# Now let me also look at the actual structure from the state summary more carefully.
# The state says there's a big region of color 4 (496 cells) at rows 32-61, cols 32-61
# That's exactly 30x30 = 900 cells... but 496? So it's not fully filled with 4.

# Let me count color 4 in that region
count_4 = 0
for r in range(32, 62):
    for c in range(32, 62):
        if grid[r][c] == 4:
            count_4 += 1
print(f"\nColor 4 count in rows 32-61, cols 32-61: {count_4}")

# Total color 4 is 496. Let me see where else color 4 appears
count_4_total = 0
for r in range(64):
    for c in range(64):
        if grid[r][c] == 4:
            count_4_total += 1
print(f"Total color 4: {count_4_total}")

# Where does color 4 appear outside the main region?
print("\nColor 4 positions outside rows 32-61 or cols 32-61:")
for r in range(64):
    for c in range(64):
        if grid[r][c] == 4 and (r < 32 or r > 61 or c < 32 or c > 61):
            print(f"  ({c},{r})")
