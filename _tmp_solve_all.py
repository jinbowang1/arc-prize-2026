import sys
from harness.env import Action, Game, action_space
from harness.percept import analyze
from harness.probe import probe_counters, probe_volatile
from harness.search import bfs_level_up

log = open('/tmp/r11l_solutions.txt', 'w')
def w(*a):
    print(*a, flush=True)
    log.write(' '.join(str(x) for x in a) + '\n'); log.flush()

game, obs = Game.make("r11l")
sp = action_space(list(obs.actions))
fullplan = []
total_steps = 0
while getattr(obs, 'level', 0) < getattr(obs, 'win_levels', 6):
    lvl = getattr(obs, 'level', 0)
    w(f"=== LEVEL {lvl} start ===")
    scene = analyze(obs.grid)
    clicks = [Action.click(c, r) for (r, c) in scene.targets]
    mask = probe_volatile(game, obs, [], clicks) & probe_counters(game, obs, clicks)
    res = bfs_level_up(game, obs, [], 6, mask, max_depth=60, max_nodes=200000, max_seconds=300)
    w(res.text())
    if not res.solved:
        w("!! could not solve level", lvl); break
    seq = []
    for a in res.seq:
        d = dict(a.data)
        seq.append((d['x'], d['y']))
        obs = game.act(a)
    total_steps += len(seq)
    w(f"LEVEL {lvl} SOLUTION ({len(seq)} steps):", seq)
    w("now level", getattr(obs,'level',None))
    fullplan.append((lvl, seq))
w("=== FULL PLAN (total steps = %d) ===" % total_steps)
for lvl, seq in fullplan:
    w(f"L{lvl}:", seq)
log.close()