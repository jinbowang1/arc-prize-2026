import sys,time; sys.path.insert(0,'probes/20260826/play_r11l')
from gp import *; from solve import *; from l3bfs_core import bfs_group
env,o=load('r11l'); gm=clone(env); total=[]; ob=None; human=None
for (r,c) in [(11,39),(59,27),(31,39)]: ob=cclick(gm,r,c)
total.append(('L1',3)); print('L1 done')
for lvl in range(2,10):
    groups,hz,wall=level_info(gm); steps=0; failed=False; clicks=[]
    groups={k:g for k,g in groups.items() if 'st' in g and 'ring' in g and 'marker' in g and 'dirwzt' not in k}
    print(f'== L{lvl}: groups', {k:(len(g['st']), center(g['ring']) if 'ring' in g else None) for k,g in groups.items()}, 'hazard',int(hz.sum()), flush=True)
    for k,g in groups.items():
        if 'ring' not in g or 'marker' not in g: print('  组',k,'缺 ring/marker'); continue
        t0=time.time(); res=bfs_group(g,gm.wiayqaumjug,gm)
        print(f'  组{k} plan={res[0] if res else None} ({time.time()-t0:.0f}s)', flush=True)
        if not res: failed=True; break
        for act in res[1]:
            pt=center(g['st'][act[1]]) if act[0]=='sel' else act[2]
            ob=cclick(gm,*pt); clicks.append(pt)
            steps+=1
        print(f'     → levels={ob.levels_completed} state={ob.state.name} strikes={gm.yledlprvvkb} 已用步数={gm._action_count}', flush=True)
        if ob.state.name=='GAME_OVER': failed=True; break
    total.append((f'L{lvl}',steps)); print(f'  CLICKS L{lvl}:',clicks, flush=True)
    if failed or ob.levels_completed<lvl: print('  ✗ 本关未过'); break
    if ob.state.name=='WIN': print('WIN'); break
print('TOTAL',total, sum(s for _,s in total))
