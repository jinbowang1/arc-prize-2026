import sys,time,heapq,itertools; sys.path.insert(0,'probes/20260826/play_r11l')
from gp import *; from solve import *
env,o=load('r11l'); gm=clone(env)
for (r,c) in [(11,39),(59,27),(31,39),(25,45),(21,8),(61,33),(9,49),(61,36),(35,45),(4,49),(48,54),(28,61)]: ob=cclick(gm,r,c)
groups,hz,wall=level_info(gm)
def bfs_group(g,sel,step=4,cap=240,maxcost=14):
    st=g['st']; mk=g['marker']; ring=g['ring']; n=len(st); rc=center(ring); ringcells=set(marker_cells(ring,*rc))
    mok=lambda c: all(0<=y<64 and 0<=x<64 and not hz[y,x] for y,x in marker_cells(mk,*c))
    mwin=lambda c: any((y,x) in ringcells for y,x in marker_cells(mk,*c))
    def sok(c):
        cy,cx=c
        return all(0<=cy-2+dy<64 and 0<=cx-2+dx<64 and not wall[cy-2+dy,cx-2+dx] for dy in range(5) for dx in range(5))
    cand=[(y,x) for y in range(3,62,step) for x in range(3,62,step) if sok((y,x))]
    # 靶心附近加细网格, 保证能精确命中
    cand+= [(y,x) for y in range(rc[0]-6,rc[0]+7) for x in range(rc[1]-6,rc[1]+7) if 0<y<64 and 0<x<64 and sok((y,x)) and (y,x) not in cand]
    start=tuple(center(s) for s in st); si=st.index(sel) if sel in st else None
    h=lambda pos: (abs(centroid(pos)[0]-rc[0])+abs(centroid(pos)[1]-rc[1]))/ (8)  # 乐观: 一次搬动重心最多挪 ~8 格
    pq=[(h(start),0,start,si,[])]; seen={}; t0=time.time(); expanded=0
    while pq:
        f,cost,pos,cur,path=heapq.heappop(pq)
        if seen.get((pos,cur),99)<=cost: continue
        seen[(pos,cur)]=cost; expanded+=1
        if mwin(centroid(pos)): return cost,path,pos
        if cost>=maxcost or time.time()-t0>cap: continue
        for i in range(n):
            c1=cost+(0 if i==cur else 1)+1
            for c in cand:
                if c==pos[i]: continue
                if any(j!=i and abs(c[0]-pos[j][0])<=2 and abs(c[1]-pos[j][1])<=2 for j in range(n)): continue
                np_=list(pos); np_[i]=c; np_=tuple(np_)
                if not mok(centroid(np_)): continue
                if seen.get((np_,i),99)<=c1: continue
                heapq.heappush(pq,(c1+h(np_),c1,np_,i,path+([('sel',i)] if i!=cur else [])+[('move',i,c)]))
    return None
for k,g in groups.items():
    t0=time.time(); res=bfs_group(g,gm.wiayqaumjug)
    print(k,'plan',res,f'{time.time()-t0:.0f}s',flush=True)
