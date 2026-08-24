from harness.env import Action, Game, action_space
from harness.percept import analyze
from harness.probe import probe_counters, probe_volatile
from harness.search import bfs_level_up

game, obs = Game.make("r11l")
sp = action_space(list(obs.actions))
scene = analyze(obs.grid)
clicks = [Action.click(c, r) for (r, c) in scene.targets]
mask = probe_volatile(game, obs, [], clicks) & probe_counters(game, obs, clicks)
res = bfs_level_up(game, obs, [], 6, mask, max_depth=40, max_nodes=60000, max_seconds=180)
print(res.text())
print("SEQ:")
for a in res.seq:
    print("  click", a.data)
# Verify by replaying
v = game.fork()
o = obs
for a in res.seq:
    o = v.act(a)
print("after replay: level", getattr(o, 'level', None), "state", getattr(o,'state',None))