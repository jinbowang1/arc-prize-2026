"""r11l 通用求解器: 读克隆体精灵, 规划'选站+搬站'序列使各组标记(重心)进靶环, 中途不碰危险块/墙."""
import sys, itertools, heapq, numpy as np
sys.path.insert(0,'probes/20260826/play_r11l')
from gp import *
def level_info(gm):
    lv=gm.current_level; groups={}; hz=np.zeros((64,64),bool); wall=np.zeros((64,64),bool)
    def paint(mask,s):
        p=s.pixels; h,w=p.shape
        for dy in range(h):
            for dx in range(w):
                y,x=s.y+dy,s.x+dx
                if 0<=y<64 and 0<=x<64 and p[dy,dx]>0: mask[y,x]=True
    for s in lv.get_sprites():
        n=s.name
        if n.startswith('defgjl'): paint(hz,s)
        elif n.startswith('wakneh'): paint(wall,s)
        elif n.startswith('roefwulewcui-'): groups.setdefault(n.split('-',1)[1],{}).setdefault('st',[]).append(s)
        elif n.startswith('flkdtg-'): groups.setdefault(n.split('-',1)[1],{})['ring']=s
        elif n.startswith('roefwu-'): groups.setdefault(n.split('-',1)[1],{})['marker']=s
    return groups,hz,wall
def center(s): return (s.y+s.height//2, s.x+s.width//2)
def marker_cells(mk, cy, cx):
    """标记精灵放在中心(cy,cx)时占的非零格"""
    p=mk.pixels; h,w=p.shape; y0,x0=cy-h//2,cx-w//2
    return [(y0+dy,x0+dx) for dy in range(h) for dx in range(w) if p[dy,dx]>0]
def centroid(pts): return (sum(p[0] for p in pts)//len(pts), sum(p[1] for p in pts)//len(pts))
def plan_group(key, g, hz, wall, selected_name, sel_center):
    st=g['st']; mk=g['marker']; ring=g['ring']; n=len(st)
    rc=center(ring); ringcells=set(marker_cells(ring,*rc))
    def marker_ok(c):
        cells=marker_cells(mk,*c)
        return all(0<=y<64 and 0<=x<64 and not hz[y,x] for y,x in cells)
    def marker_win(c):
        return any((y,x) in ringcells for y,x in marker_cells(mk,*c))
    def station_ok(c):
        cy,cx=c; h,w=st[0].height,st[0].width
        for dy in range(h):
            for dx in range(w):
                y,x=cy-h//2+dy,cx-w//2+dx
                if not(0<=y<64 and 0<=x<64) or wall[y,x]: return False
        return True
    cand=[(y,x) for y in range(2,62,2) for x in range(2,62,2) if station_ok((y,x))]
    start=tuple(center(s) for s in st); sel=None
    for i,s in enumerate(st):
        if s is selected_name: sel=i
    # Dijkstra: state=(positions, selected idx), cost=actions
    pq=[(0,start,sel,[])]; seen={}
    while pq:
        cost,pos,si,path=heapq.heappop(pq)
        if (pos,si) in seen and seen[(pos,si)]<=cost: continue
        seen[(pos,si)]=cost
        if marker_win(centroid(pos)): return cost,path,pos
        if cost>12: continue
        for i in range(n):
            c1=cost+(0 if i==si else 1)
            for c in cand:
                if c==pos[i]: continue
                np_=list(pos); np_[i]=c; np_=tuple(np_)
                if not marker_ok(centroid(np_)): continue
                heapq.heappush(pq,(c1+1,np_,i,path+[('sel',i) if i!=si else None,('move',i,c)]))
    return None

def plan_group2(key, g, hz, wall, selected, max_move=3):
    st=g['st']; mk=g['marker']; ring=g['ring']; n=len(st)
    rc=center(ring); ringcells=set(marker_cells(ring,*rc))
    def marker_ok(c): return all(0<=y<64 and 0<=x<64 and not hz[y,x] for y,x in marker_cells(mk,*c))
    def marker_win(c): return any((y,x) in ringcells for y,x in marker_cells(mk,*c))
    def station_ok(c):
        cy,cx=c; h,w=st[0].height,st[0].width
        for dy in range(h):
            for dx in range(w):
                y,x=cy-h//2+dy,cx-w//2+dx
                if not(0<=y<64 and 0<=x<64) or wall[y,x]: return False
        return True
    cand=[(y,x) for y in range(2,62,1) for x in range(2,62,1) if station_ok((y,x))]
    candset=set(cand)
    start=[center(s) for s in st]; sel=st.index(selected) if selected in st else None
    if marker_win(centroid(start)): return 0,[],start
    best=None
    for k in range(1,min(max_move,n)+1):
        for movers in itertools.combinations(range(n),k):
            fixed_sum=[sum(start[i][d] for i in range(n) if i not in movers) for d in (0,1)]
            # 枚举前 k-1 个搬动站的落点, 最后一个由重心约束解出(±2 容差)
            free=movers[:-1]; last=movers[-1]
            for combo in itertools.product(cand if k>1 else [None], repeat=k-1):
                if k>1 and len(set(combo))<len(combo): continue
                s0=fixed_sum[0]+sum(c[0] for c in combo if c); s1=fixed_sum[1]+sum(c[1] for c in combo if c)
                for ty in range(rc[0]-2,rc[0]+3):
                    for tx in range(rc[1]-2,rc[1]+3):
                        py,px=n*ty-s0+ (n-1 if False else 0), n*tx-s1
                        # centroid 用整除, 试 py..py+n-1 微调
                        for oy in range(n):
                            for ox in range(n):
                                P=(py+oy,px+ox)
                                if P not in candset: continue
                                final=list(start)
                                for i,c in zip(free,combo): final[i]=c
                                final[last]=P
                                if not marker_win(centroid(final)): continue
                                # 落点不能压在别的站点身上(点到站点=选中而非搬动), 含起始与终态布局
                                clash=False
                                for i in movers:
                                    for j in range(n):
                                        if j==i: continue
                                        for other in (start[j],final[j]):
                                            if abs(final[i][0]-other[0])<=2 and abs(final[i][1]-other[1])<=2: clash=True
                                if clash: continue
                                # 排序: 找一个顺序使中途重心不碰危险
                                for order in itertools.permutations(movers):
                                    pos=list(start); ok=True; path=[]; cur=sel; cost=0
                                    for i in order:
                                        pos[i]=final[i]
                                        if not marker_ok(centroid(pos)): ok=False; break
                                        if cur!=i: path.append(('sel',i)); cost+=1; cur=i
                                        path.append(('move',i,final[i])); cost+=1
                                    if ok and (best is None or cost<best[0]): best=(cost,path,final)
                if best and best[0]<=k*2-1: break
            if best: break
        if best: break
    return best
