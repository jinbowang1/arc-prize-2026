import sys,time,itertools; sys.path.insert(0,'probes/20260826/play_r11l')
from gp import *; from solve import *
env,o=load('r11l'); gm=clone(env)
for (r,c) in [(11,39),(59,27),(31,39),(25,45),(21,8),(61,33),(9,49),(61,36),(35,45),(4,49),(48,54),(28,61)]: ob=cclick(gm,r,c)
groups,hz,wall=level_info(gm)
def solve_group(g,sel,step=3,rowmin=0,colmin=0,tries_cap=10**7):
    st=g['st']; mk=g['marker']; ring=g['ring']; n=len(st); rc=center(ring); ringcells=set(marker_cells(ring,*rc))
    mok=lambda c: all(0<=y<64 and 0<=x<64 and not hz[y,x] for y,x in marker_cells(mk,*c))
    mwin=lambda c: any((y,x) in ringcells for y,x in marker_cells(mk,*c))
    def sok(c):
        cy,cx=c
        return all(0<=cy-2+dy<64 and 0<=cx-2+dx<64 and not wall[cy-2+dy,cx-2+dx] for dy in range(5) for dx in range(5))
    cand=[(y,x) for y in range(max(2,rowmin),62,step) for x in range(max(2,colmin),62,step) if sok((y,x))]; cs=set((y,x) for y in range(2,62) for x in range(2,62) if sok((y,x)))
    start=[center(s) for s in st]; si=st.index(sel) if sel in st else None; best=None; t0=time.time()
    for combo in itertools.product(cand,repeat=n-1):
        if len(set(combo))<n-1: continue
        s0=sum(c[0] for c in combo); s1=sum(c[1] for c in combo)
        for ty in range(rc[0]-1,rc[0]+2):
            for tx in range(rc[1]-1,rc[1]+2):
                P=(n*ty-s0+1, n*tx-s1+1)
                if P not in cs: continue
                final=list(combo)+[P]
                if any(abs(final[i][0]-final[j][0])<=2 and abs(final[i][1]-final[j][1])<=2 for i in range(n) for j in range(i)): continue
                if not mwin(centroid(final)): continue
                for order in itertools.permutations(range(n)):
                    pos=list(start); ok=True; path=[]; cur=si; cost=0
                    for i in order:
                        # 落点不能压在当时其它站上
                        if any(j!=i and abs(final[i][0]-pos[j][0])<=2 and abs(final[i][1]-pos[j][1])<=2 for j in range(n)): ok=False; break
                        pos[i]=final[i]
                        if not mok(centroid(pos)): ok=False; break
                        if cur!=i: path.append(('sel',i)); cost+=1; cur=i
                        path.append(('move',i,final[i])); cost+=1
                    if ok and (best is None or cost<best[0]): best=(cost,path,final)
        if time.time()-t0>500: print('time cap'); break
    return best
for k,g in groups.items():
    t0=time.time(); res=solve_group(g,gm.wiayqaumjug,step=3,rowmin=38 if len(g['st'])==4 else 0)
    print(k,'plan',res,f'{time.time()-t0:.0f}s',flush=True)
