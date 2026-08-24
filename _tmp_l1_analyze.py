import numpy as np
from harness.env import Action, Game, action_space
from harness.percept import analyze

game, obs = Game.make("r11l")
for (cx,cy) in [(38,18),(28,59),(41,19)]:
    obs = game.act(Action.click(cx,cy))
print("level", getattr(obs,'level',None))
print("actions", getattr(obs,'actions',None))

# Report color positions: find center of colored components of interest
g=np.array(obs.grid)
# gold color 6 (the gold centers), color 0,1,3,15
import sys
def centroid_of_color(c):
    pts=np.argwhere(g==c)
    if len(pts)==0: return None
    ys=[p[0] for p in pts]; xs=[p[1] for p in pts]
    return (min(ys),max(ys),min(xs),max(xs),len(pts))
for c in [0,1,3,6,15]:
    r=centroid_of_color(c)
    print(f"color {c}: rows/cols/count {r}")
print("scene.targets count", len(analyze(obs.grid).targets))