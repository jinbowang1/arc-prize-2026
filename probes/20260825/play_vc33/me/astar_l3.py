import sys,json,hashlib,time,heapq,itertools; sys.path.insert(0,'probes/20260825/play_vc33/me')
from gp import *
env,o=load('vc33')
P='probes/20260825/play_vc33/me/solutions.json'; sol=json.load(open(P))
for k in sorted(sol, key=lambda s:int(s[1:])):
    for a,x,y in sol[k]: o=act(env,6,x,y)
lvl=o.levels_completed; g0=grid(o)
cs,bg=comps(g0); B=[c['center'] for c in cs if c['color']==9]
# 零件(3格标记)与门(2格)配对
def h(g):
    s=0
    for col in (14,15,11):
        cs,_=comps(np.where(g==col,col,3),bg=3)
        if len(cs)<2: continue
        cs=sorted(cs,key=lambda c:c['n']); gt=cs[0]; pc=cs[-1]   # 门=最小块, 零件=最大块
        s+=abs(pc['center'][0]-gt['center'][0])   # 只算行距: 门和零件在不同竖井, 列差不可消
    return s/3
def clonegame(gm):
    clean=gm._clean_levels; gm._clean_levels=None; g2=copy.deepcopy(gm); gm._clean_levels=clean; g2._clean_levels=clean; return g2
root=clone(env); cnt=itertools.count()
pq=[(h(g0),0,next(cnt),root,[])]; seen={hashlib.md5(g0.tobytes()).hexdigest()}; t0=time.time(); n=0; found=None
while pq and n<int(sys.argv[1]) if len(sys.argv)>1 else 150000:
    f,gcost,_,gm,seq=heapq.heappop(pq)
    for (y,x) in B:
        g2=clonegame(gm); r=g2.perform_action(ActionInput(id=ACTS[6],data={'x':x,'y':y}),raw=True); n+=1
        if r.levels_completed>lvl or r.state.name=='WIN': found=seq+[(y,x)]; break
        if not r.frame or r.state.name=='GAME_OVER': continue
        g1=np.array(r.frame[-1]); k=hashlib.md5(g1.tobytes()).hexdigest()
        if k in seen: continue
        seen.add(k); heapq.heappush(pq,(gcost+1+h(g1),gcost+1,next(cnt),g2,seq+[(y,x)]))
    if found: break
    if n%20000<len(B): print(f'  节点{n} 状态{len(seen)} 队首f={f:.1f} g={gcost} {time.time()-t0:.0f}s',flush=True)
if found:
    print(f'L{lvl+1} 解出 {len(found)} 步 (节点{n}, {time.time()-t0:.0f}s)')
    seq=[[6,x,y] for y,x in found]
    for a,x,y in seq: o=act(env,6,x,y)
    sol[f'L{lvl+1}']=seq; json.dump(sol,open(P,'w')); print('真机 level',o.levels_completed,'state',o.state.name)
else: print('未解出', n, len(seen))
