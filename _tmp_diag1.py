import time
from harness.env import Action, Game, action_space
from harness.percept import analyze, click_targets
from harness.probe import probe_counters, probe_volatile

game, obs = Game.make("r11l")
# advance to level 1 using L0 solution
for (cx,cy) in [(38,18),(28,59),(41,19)]:
    obs = game.act(Action.click(cx,cy))
print("now level", getattr(obs,'level',None))
scene = analyze(obs.grid)
cts = click_targets(__import__('numpy').array(obs.grid))
print("analyze targets:", len(scene.targets), "click_targets:", len(cts))
clicks = [Action.click(c, r) for (r, c) in scene.targets]
t0=time.time(); mask = probe_volatile(game, obs, [], clicks); t1=time.time()
print("probe_volatile", round(t1-t0,1),"s")
t0=time.time(); mask2 = probe_counters(game, obs, clicks); t1=time.time()
print("probe_counters", round(t1-t0,1),"s")
