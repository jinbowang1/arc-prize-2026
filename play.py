import arc_agi
from arcengine import GameAction

arc = arc_agi.Arcade()
env = arc.make("ls20")

for _ in range(10):
    env.step(GameAction.ACTION1)

print(arc.get_scorecard())
