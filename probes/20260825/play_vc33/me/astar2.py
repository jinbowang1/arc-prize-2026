"""vc33 A* v2 (按钮+闸门+冲动画): h = Σ船 (|船行-门行|/2 + 8·[船不在门管道相邻舱])"""
import sys,json,hashlib,time,heapq,itertools; sys.path.insert(0,'probes/20260825/play_vc33/me')
from gp import *
LIMIT=int(sys.argv[1]) if len(sys.argv)>1 else 300000
env,o=load('vc33')
P='probes/20260825/play_vc33/me/solutions.json'; sol=json.load(open(P))
for k in sorted(sol, key=lambda s:int(s[1:])):
    for a,x,y in sol[k]: o=act(env,6,x,y)
lvl=o.levels_completed; g0=grid(o); print('level',lvl)
def cg(gm):
    c=gm._clean_levels; gm._clean_levels=None; g2=copy.deepcopy(gm); gm._clean_levels=c; g2._clean_levels=c; return g2
def key(g): return hashlib.md5(g[1:].tobytes()).hexdigest()
PIPES=[(22,23),(40,41)]
def h(g):
    s=0
    for col in (11,14,15):
        cs,_=comps(np.where(g[1:]==col,col,3),bg=3)
        if len(cs)<2: continue
        cs=sorted(cs,key=lambda c:c['n']); gt=cs[0]; bt=cs[-1]
        gy=gt['center'][0]+1; gx=gt['center'][1]; by=bt['center'][0]+1; bx=bt['center'][1]
        s+=abs(by-gy)/2
        # 门所在管道
        pipe=[p for p in PIPES if p[0]<=gx<=p[1]]
        if pipe:
            p=pipe[0]; adj = (p[0]-16<=bx<=p[0]-1) or (p[1]+1<=bx<=p[1]+16)
            if not adj: s+=8
    return s
cs,bg=comps(g0); acts=[c['center'] for c in cs if c['color']==9]+[c['center'] for c in cs if c['color']==1]
cnt=itertools.count(); pq=[(h(g0),0,next(cnt),clone(env),[],g0)]; seen={key(g0)}; n=0; t0=time.time(); found=None
while pq and n<LIMIT:
    f,gc,_,gm,seq,gprev=heapq.heappop(pq)
    for (y,x) in acts:
        g2=cg(gm); r=g2.perform_action(ActionInput(id=ACTS[6],data={'x':x,'y':y}),raw=True); n+=1; steps=[(y,x)]
        if r.levels_completed>lvl or r.state.name=='WIN': found=seq+steps; break
        if not r.frame or r.state.name!='NOT_FINISHED': continue
        g1=np.array(r.frame[-1])
        if key(g1)==key(gprev):   # 画面没变 → 可能是动画, 用背景点击冲
            for i in range(40):
                r=g2.perform_action(ActionInput(id=ACTS[6],data={'x':1,'y':10}),raw=True); steps.append((10,1))
                if r.levels_completed>lvl or r.state.name=='WIN': found=seq+steps; break
                if not r.frame or r.state.name!='NOT_FINISHED': break
                gn=np.array(r.frame[-1])
                if key(gn)==key(g1) and i>0: break
                g1=gn
            if found: break
            if not r.frame or r.state.name!='NOT_FINISHED' or key(g1)==key(gprev): continue
        k=key(g1)
        if k in seen: continue
        seen.add(k); heapq.heappush(pq,(gc+len(steps)+h(g1),gc+len(steps),next(cnt),g2,seq+steps,g1))
    if found: break
    if n%20000<len(acts): print(f'  节点{n} 状态{len(seen)} 队首f={f:.1f} g={gc} {time.time()-t0:.0f}s',flush=True)
if found:
    print(f'L{lvl+1} 解出 {len(found)} 步 节点{n} {time.time()-t0:.0f}s')
    seq=[[6,x,y] for y,x in found]
    for a,x,y in seq: o=act(env,6,x,y)
    print('真机 level',o.levels_completed,'state',o.state.name)
    if o.levels_completed>lvl: sol[f'L{lvl+1}']=seq; json.dump(sol,open(P,'w'))
else: print('未解出',n,len(seen))
