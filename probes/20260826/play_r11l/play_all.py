import sys,time,json; sys.path.insert(0,'probes/20260826/play_r11l')
from gp import *; from solve import *
env,o=load('r11l'); gm=clone(env); total=[]; ob=None
# L1 已知 3 步解
for (r,c) in [(11,39),(59,27),(31,39)]: ob=cclick(gm,r,c)
total.append(('L1',3)); print('L1 done levels=',ob.levels_completed)
for lvl in range(2,10):
    groups,hz,wall=level_info(gm); sel=gm.wiayqaumjug; steps=0
    print(f'== L{lvl}: groups', {k:(len(g['st']), center(g['ring']) if 'ring' in g else None) for k,g in groups.items()}, 'hazard cells',int(hz.sum()))
    for k,g in groups.items():
        if 'ring' not in g or 'marker' not in g: print('  组',k,'缺 ring/marker, 跳过'); continue
        t0=time.time(); res=plan_group2(k,g,hz,wall,gm.wiayqaumjug)
        print(f'  组{k} plan={res[0] if res else None} ({time.time()-t0:.0f}s):', res[1] if res else None)
        if not res: break
        for act in res[1]:
            if act is None: continue
            if act[0]=='sel': i=act[1]; ob=cclick(gm,*center(g['st'][i])); steps+=1
            else: i,P=act[1],act[2]; ob=cclick(gm,*P); steps+=1
            print(f'     {act} → levels={ob.levels_completed} state={ob.state} strikes={gm.yledlprvvkb} 站中心={[center(s) for s in g["st"]]}')
        if ob.state.name=='GAME_OVER': break
    total.append((f'L{lvl}',steps))
    if ob.levels_completed<lvl: print('  ✗ 本关未过'); break
    if ob.state.name in ('WIN',): print('WIN'); break
print('TOTAL',total, sum(s for _,s in total))
