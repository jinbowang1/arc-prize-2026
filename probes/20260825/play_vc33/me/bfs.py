"""vc33 逐关 BFS: 按钮=9 色连通块中心; 状态按画面去重; 每关最多 N 节点."""
import sys,json,hashlib,time,collections; sys.path.insert(0,'probes/20260825/play_vc33/me')
from gp import *
LIMIT=int(sys.argv[1]) if len(sys.argv)>1 else 40000
env,o=load('vc33')
P='probes/20260825/play_vc33/me/solutions.json'
sol=json.load(open(P)) if os.path.exists(P) else {}
for k in sorted(sol, key=lambda s:int(s[1:])):
    for a,x,y in sol[k]: o=act(env,6,x,y)
print('从 level',o.levels_completed,'开始; state',o.state.name)
def clonegame(gm):
    clean=gm._clean_levels; gm._clean_levels=None; g2=copy.deepcopy(gm); gm._clean_levels=clean; g2._clean_levels=clean; return g2
while o.state.name!='WIN':
    lvl=o.levels_completed; g0=grid(o)
    cs,bg=comps(g0); buttons=[c['center'] for c in cs if c['color']==9]
    print(f'L{lvl+1}: 按钮 {len(buttons)} 个 {buttons}')
    root=clone(env); start=hashlib.md5(g0.tobytes()).hexdigest()
    q=collections.deque([(root,[])]); seen={start}; found=None; t0=time.time(); n=0
    while q and found is None and n<LIMIT:
        gm,seq=q.popleft()
        for (y,x) in buttons:
            g2=clonegame(gm); r=g2.perform_action(ActionInput(id=ACTS[6],data={'x':x,'y':y}),raw=True); n+=1
            if r.levels_completed>lvl or r.state.name=='WIN': found=seq+[(y,x)]; break
            g1=np.array(r.frame[-1]); h=hashlib.md5(g1.tobytes()).hexdigest()
            if h in seen: continue
            seen.add(h); q.append((g2,seq+[(y,x)]))
    if not found: print(f'L{lvl+1} 未解出: 节点{n} 状态{len(seen)} {time.time()-t0:.0f}s'); break
    print(f'L{lvl+1} 解出 {len(found)} 步 (节点{n}, 状态{len(seen)}, {time.time()-t0:.0f}s)')
    seq=[[6,x,y] for y,x in found]
    for a,x,y in seq: o=act(env,6,x,y)
    sol[f'L{lvl+1}']=seq; json.dump(sol,open(P,'w'))
    print('  真机 level',o.levels_completed,'state',o.state.name)
print('结束 state',o.state.name,'levels',o.levels_completed,'各关步数',{k:len(v) for k,v in sol.items()})
