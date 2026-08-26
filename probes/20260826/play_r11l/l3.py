import sys,time; sys.path.insert(0,'probes/20260826/play_r11l')
from gp import *; from solve import *
import solve
env,o=load('r11l'); gm=clone(env)
for (r,c) in [(11,39),(59,27),(31,39),(25,45),(21,8),(61,33),(9,49),(61,36),(35,45),(4,49),(48,54),(28,61)]: ob=cclick(gm,r,c)
print('levels',ob.levels_completed); g=np.array(ob.frame[-1]); show(g)
groups,hz,wall=level_info(gm)
for k,gr in groups.items(): print(k,'st',[center(s) for s in gr['st']],'ring',center(gr['ring']),'marker',center(gr['marker']), 'sel' , gr['st'].index(gm.wiayqaumjug) if gm.wiayqaumjug in gr['st'] else None)
