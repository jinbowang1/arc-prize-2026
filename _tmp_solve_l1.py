import time
from harness.env import Action, Game, action_space
from harness.percept import analyze
from harness.probe import probe_counters, probe_volatile
from harness.search import bfs_level_up

log = open('/tmp/r11l_l1.txt', 'w')
def w(*a):
    line=' '.join(str(x) for x in a); print(line, flush=True); log.write(line+'\n'); log.flush()

game, obs = Game.make("r11l")
# L0 solution to get to level 1
for (cx,cy) in [(38,18),(28,59),(41,19)]:
    obs = game.act(Action.click(cx,cy))
w("at level", getattr(obs,'level',None))
scene = analyze(obs.grid)
clicks = [Action.click(c, r) for (r, c) in scene.targets]
mask = probe_volatile(game, obs, [], clicks) & probe_counters(game, obs, clicks)
w("targets", len(clicks))
t0=time.time()
res = bfs_level_up(game, obs, [], 6, mask, max_depth=80, max_nodes=1500000, max_seconds=600)
w("result", res.text())
if res.solved:
    w("SOLUTION:")
    for a in res.seq:
        w("  click", dict(a.data))
else:
    w("no solution found")
log.close()