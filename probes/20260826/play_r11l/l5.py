import sys,time,heapq,numpy as np; sys.path.insert(0,'probes/20260826/play_r11l')
from upto import *; from solve import center, centroid
gm,ob=upto(5); lv=gm.current_level
sprs=lv.get_sprites(); walls=[s for s in sprs if s.name.startswith('wakneh')]
puk={s.name.split('hawffu')[1]:s for s in sprs if s.name.startswith('puukul')}
rings={s.name.split('-',1)[1]:s for s in sprs if s.name.startswith('flkdtg') and 'dirwzt' not in s.name}
groups={}
for s in sprs:
    if s.name.startswith('roefwulewcui-'): groups.setdefault(s.name.split('-',1)[1],{}).setdefault('st',[]).append(s)
    elif s.name.startswith('roefwu-'): groups.setdefault(s.name.split('-',1)[1],{})['marker']=s
def ringcolors(r): return sorted({int(v) for v in np.unique(r.pixels) if v>0})
# 组→靶环: 按颜色匹配 puukul 名字: ring 'blxuubrengnt' = puukul blxuub(9)+rengnt(8)
tasks={'whkxtx':('blxuubrengnt',['rengnt','blxuub']), 'whkxtx-2':('yeogyfgrhcew',['yeogyf','grhcew'])}
def masks(g, ringname):
    st=g['st'][0]; mk=g['marker']; ring=rings[ringname]; memo={}
    class M:
        def __init__(s,f): s.f=f; s.c={}
        def __getitem__(s,k):
            if k not in s.c: s.c[k]=s.f(*k)
            return s.c[k]
    def _st(cy,cx):
        sx,sy=st.x,st.y; st.set_position(cx-2,cy-2); r=not any(st.collides_with(w) for w in walls); st.set_position(sx,sy); return r
    def _mk(cy,cx,other):
        mx,my=mk.x,mk.y; mk.set_position(cx-2,cy-2); r=mk.collides_with(other); mk.set_position(mx,my); return r
    st_ok=M(_st); win=M(lambda cy,cx:_mk(cy,cx,ring))
    ov={k:M(lambda cy,cx,p=p:_mk(cy,cx,p)) for k,p in puk.items()}
    full={k:M(lambda cy,cx,p=p:(p.x==cx-2 and p.y==cy-2)) for k,p in puk.items()}
    return st_ok,win,ov,full
def plan(gkey, ringname, colors, cap=300):
    g=groups[gkey]; st=g['st']; n=len(st); st_ok,win,ov,full=masks(g,ringname)
    A,B=colors; others=[k for k in puk if k not in colors]
    rc=center(rings[ringname]); pa=center(puk[A]); pb=center(puk[B])
    cand=[(y,x) for y in range(3,62,4) for x in range(3,62,4) if st_ok[y,x]]
    for (ty,tx) in (rc,pa,pb):
        cand+=[(y,x) for y in range(ty-6,ty+7) for x in range(tx-6,tx+7) if 2<=y<62 and 2<=x<62 and (y,x) not in cand and st_ok[y,x]]
    start=tuple(center(s) for s in st); si=st.index(gm.wiayqaumjug) if gm.wiayqaumjug in st else None
    goal=[pa,pb,rc]
    h=lambda pos,stg: (abs(centroid(pos)[0]-goal[stg][0])+abs(centroid(pos)[1]-goal[stg][1]))/8
    pq=[(h(start,0),0,start,si,0,[])]; seen={}; t0=time.time()
    while pq:
        f,cost,pos,cur,stg,path=heapq.heappop(pq)
        if seen.get((pos,cur,stg),99)<=cost: continue
        seen[(pos,cur,stg)]=cost
        if stg==3: return cost,path,pos
        if cost>=24 or time.time()-t0>cap: continue
        for i in range(n):
            c1=cost+(0 if i==cur else 1)+1
            for c in cand:
                if c==pos[i] or any(j!=i and abs(c[0]-pos[j][0])<=2 and abs(c[1]-pos[j][1])<=2 for j in range(n)): continue
                np_=list(pos); np_[i]=c; np_=tuple(np_); cc=centroid(np_)
                if any(ov[o][cc] for o in others): continue      # 别吃别组的色
                ns=stg
                if stg==0:
                    if ov[B][cc]: continue
                    if full[A][cc]: ns=1
                    elif ov[A][cc]: continue
                elif stg==1:
                    if ov[A][cc]: continue  # A 已被吃掉, 实际不会有; 保守
                    if ov[B][cc] and not full[B][cc]: ns=2
                    elif full[B][cc]: continue
                elif stg==2:
                    if win[cc]: ns=3
                if seen.get((np_,i,ns),99)<=c1: continue
                heapq.heappush(pq,(c1+h(np_,min(ns,2)),c1,np_,i,ns,path+([('sel',i)] if i!=cur else [])+[('move',i,c)]))
    return None
if __name__=='__main__':
    allclicks=[]
    for gkey,(ringname,colors) in tasks.items():
        t0=time.time(); res=plan(gkey,ringname,colors); print(gkey,'plan',res[0] if res else None,f'{time.time()-t0:.0f}s',res[1] if res else '',flush=True)
        if not res: break
        g=groups[gkey]
        for a in res[1]:
            pt=center(g['st'][a[1]]) if a[0]=='sel' else a[2]; ob=cclick(gm,*pt); allclicks.append(pt)
            mk=g['marker']; print('   ',a,'标记色',sorted({int(v) for v in np.unique(mk.pixels) if v>0}),'levels',ob.levels_completed,'state',ob.state.name,'步',gm._action_count)
    print('CLICKS L5:',allclicks)
