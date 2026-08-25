import sys,json,hashlib,collections,time; sys.path.insert(0,'probes/20260825/play_vc33/me')
from gp import *
env,o=load('vc33')
sol=json.load(open('probes/20260825/play_vc33/me/solutions.json'))
for k in ['L1','L2','L3','L4','L5','L6']:
    for a,x,y in sol[k]: o=act(env,6,x,y)
g0=grid(o)
def cg(gm):
    c=gm._clean_levels; gm._clean_levels=None; g2=copy.deepcopy(gm); gm._clean_levels=c; g2._clean_levels=c; return g2
def key(g): return hashlib.md5(g[1:].tobytes()).hexdigest()
def chamber_of(y,x):
    r='上' if y<=29 else '下'; cidx='左' if x<=21 else ('中' if x<=39 else '右'); return r+cidx
def boats(g):
    d={}
    for col,n in [(11,'b'),(14,'e'),(15,'f')]:
        cs,_=comps(np.where(g==col,col,3),bg=3); cs=sorted(cs,key=lambda c:c['n'])
        if cs: y,x=cs[-1]['center']; d[n]=(chamber_of(y,x),y)
    return d
cs,bg=comps(g0); acts=[c['center'] for c in cs if c['color']==9]+[c['center'] for c in cs if c['color']==1]
q=collections.deque([(clone(env),[],g0)]); seen={key(g0)}; n=0; t0=time.time(); reach=collections.defaultdict(set); found=None; LIMIT=int(sys.argv[1]) if len(sys.argv)>1 else 250000
while q and n<LIMIT and not found:
    gm,seq,g=q.popleft()
    for (y,x) in acts:
        g2=cg(gm); r=g2.perform_action(ActionInput(id=ACTS[6],data={'x':x,'y':y}),raw=True); n+=1
        if r.levels_completed>6 or r.state.name=='WIN': found=seq+[(y,x)]; break
        if not r.frame or r.state.name!='NOT_FINISHED': continue
        g1=np.array(r.frame[-1]); h=key(g1)
        if h in seen: continue
        seen.add(h); q.append((g2,seq+[(y,x)],g1))
        for b,(ch,yy) in boats(g1).items(): reach[b].add((ch,yy))
    if n%50000<len(acts): print(f'  节点{n} 状态{len(seen)} 深度{len(seq)} {time.time()-t0:.0f}s',flush=True)
print('结束 节点',n,'状态',len(seen),'found',found)
for k,v in reach.items(): print(k,'可达舱:',sorted(set(c for c,_ in v)),'行范围:',{c:(min(y for cc,y in v if cc==c),max(y for cc,y in v if cc==c)) for c in set(c for c,_ in v)})
if found:
    seq=[[6,x,y] for y,x in found]
    for a,x,y in seq: o=act(env,6,x,y)
    print('真机',o.levels_completed,o.state.name); sol['L7']=seq; json.dump(sol,open('probes/20260825/play_vc33/me/solutions.json','w'))
