import sys,time,heapq,itertools, numpy as np
sys.path.insert(0,'probes/20260826/play_r11l')
from gp import *; from solve import level_info, center, centroid
def engine_masks(gm, g):
    """用引擎碰撞预计算: st_ok[y,x] 站中心可放; mk_ok[y,x] 标记中心不碰危险; mk_win[y,x] 标记中心进环"""
    lv=gm.current_level; walls=[s for s in lv.get_sprites() if s.name.startswith('wakneh')]; hzs=[s for s in lv.get_sprites() if s.name.startswith('defgjl')]
    st=g['st'][0]; mk=g['marker']; ring=g['ring']
    st_ok=np.zeros((64,64),bool); mk_ok=np.zeros((64,64),bool); mk_win=np.zeros((64,64),bool)
    sx,sy=st.x,st.y; mx,my=mk.x,mk.y
    for cy in range(2,62):
        for cx in range(2,62):
            st.set_position(cx-st.width//2, cy-st.height//2); st_ok[cy,cx]=not any(st.collides_with(w) for w in walls)
            mk.set_position(cx-mk.width//2, cy-mk.height//2); mk_ok[cy,cx]=not any(mk.collides_with(h) for h in hzs); mk_win[cy,cx]=mk.collides_with(ring)
    st.set_position(sx,sy); mk.set_position(mx,my)
    return st_ok,mk_ok,mk_win
def bfs_group(g,sel,gm,step=4,cap=240,maxcost=14):
    st=g['st']; n=len(st); rc=center(g['ring'])
    st_ok,mk_ok,mk_win=engine_masks(gm,g)
    cand=[(y,x) for y in range(3,62,step) for x in range(3,62,step) if st_ok[y,x]]
    cand+=[(y,x) for y in range(rc[0]-6,rc[0]+7) for x in range(rc[1]-6,rc[1]+7) if 2<=y<62 and 2<=x<62 and st_ok[y,x] and (y,x) not in cand]
    start=tuple(center(s) for s in st); si=st.index(sel) if sel in st else None
    h=lambda pos: (abs(centroid(pos)[0]-rc[0])+abs(centroid(pos)[1]-rc[1]))/8
    pq=[(h(start),0,start,si,[])]; seen={}; t0=time.time()
    while pq:
        f,cost,pos,cur,path=heapq.heappop(pq)
        if seen.get((pos,cur),99)<=cost: continue
        seen[(pos,cur)]=cost
        c0=centroid(pos)
        if mk_win[c0]: return cost,path,pos
        if cost>=maxcost or time.time()-t0>cap: continue
        for i in range(n):
            c1=cost+(0 if i==cur else 1)+1
            for c in cand:
                if c==pos[i] or any(j!=i and abs(c[0]-pos[j][0])<=2 and abs(c[1]-pos[j][1])<=2 for j in range(n)): continue
                np_=list(pos); np_[i]=c; np_=tuple(np_); cc=centroid(np_)
                if not mk_ok[cc]: continue
                if seen.get((np_,i),99)<=c1: continue
                heapq.heappush(pq,(c1+h(np_),c1,np_,i,path+([('sel',i)] if i!=cur else [])+[('move',i,c)]))
    return None
