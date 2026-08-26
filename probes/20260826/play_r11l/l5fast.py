import sys,time,heapq,numpy as np; sys.path.insert(0,'probes/20260826/play_r11l')
from upto import *; from solve import center, centroid
def shape_mask(s):
    p=np.array(s.pixels); return [(dy-s.height//2,dx-s.width//2) for dy in range(p.shape[0]) for dx in range(p.shape[1]) if p[dy,dx]>0]
def cells(mask_offsets, cy, cx): return [(cy+dy,cx+dx) for dy,dx in mask_offsets]
def solve_level(gm, ob, cap=240, maxcost=26):
    lv=gm.current_level; sprs=lv.get_sprites(); g=np.array(ob.frame[-1])
    wall=(g==2); hz=(g==10)
    puk={s.name.split('hawffu')[1]:s for s in sprs if s.name.startswith('puukul')}
    rings={s.name.split('-',1)[1]:s for s in sprs if s.name.startswith('flkdtg') and 'dirwzt' not in s.name}
    groups={}
    for s in sprs:
        if s.name.startswith('roefwulewcui-'): groups.setdefault(s.name.split('-',1)[1],{}).setdefault('st',[]).append(s)
        elif s.name.startswith('roefwu-'): groups.setdefault(s.name.split('-',1)[1],{})['marker']=s
    groups={k:v for k,v in groups.items() if 'st' in v and 'marker' in v}
    def rcols(sp): return sorted({int(v) for v in np.unique(sp.pixels) if v>0})
    # 任务分配: 有色标记 → 同 key 的环; 空白标记(whkxtx) → 颜色由 puukul 名拼出的环
    tasks=[]
    used=set()
    for k,gr in groups.items():
        mc=rcols(gr['marker'])
        if k in rings: tasks.append((k,k,[])); used.add(k); continue
        # 空白: 找一个未用的环, 其名字可拆成两个 puukul key
        for rk,r in rings.items():
            if rk in used: continue
            parts=[pk for pk in puk if pk in rk]
            if len(parts)==2 and rcols(r)==sorted(rcols(puk[parts[0]])+rcols(puk[parts[1]])):
                tasks.append((k,rk,parts)); used.add(rk); break
    print('  tasks',tasks)
    allclicks=[]
    for gkey,rk,colors in tasks:
        gr=groups[gkey]; st=gr['st']; mk=gr['marker']; ring=rings[rk]; n=len(st)
        smask=shape_mask(st[0]); mmask=shape_mask(mk); rcells=set(cells(shape_mask(ring),*center(ring)))
        pcells={pk:set(cells(shape_mask(p),*center(p))) for pk,p in puk.items()}
        inb=lambda cs: all(0<=y<64 and 0<=x<64 for y,x in cs)
        st_ok=np.zeros((64,64),bool); mk_ok=np.zeros((64,64),bool); win=np.zeros((64,64),bool); ov={pk:np.zeros((64,64),bool) for pk in puk}
        for cy in range(2,62):
            for cx in range(2,62):
                sc=cells(smask,cy,cx); st_ok[cy,cx]=inb(sc) and not any(wall[y,x] for y,x in sc)
                mc_=cells(mmask,cy,cx); mk_ok[cy,cx]=inb(mc_) and not any(hz[y,x] for y,x in mc_)
                win[cy,cx]=any((y,x) in rcells for y,x in mc_)
                for pk in puk: ov[pk][cy,cx]=any((y,x) in pcells[pk] for y,x in mc_)
        full={pk:(lambda c,p=p: c==center(p)) for pk,p in puk.items()}
        others=[pk for pk in puk if pk not in colors]
        rc=center(ring); goals=[center(puk[c]) for c in colors]+[rc]; nst=len(colors)
        cand=[(y,x) for y in range(3,62,4) for x in range(3,62,4) if st_ok[y,x]]
        for (ty,tx) in goals: cand+=[(y,x) for y in range(ty-6,ty+7) for x in range(tx-6,tx+7) if 2<=y<62 and 2<=x<62 and st_ok[y,x] and (y,x) not in cand]
        start=tuple(center(s) for s in st); si=st.index(gm.wiayqaumjug) if gm.wiayqaumjug in st else None
        h=lambda pos,stg: (abs(centroid(pos)[0]-goals[stg][0])+abs(centroid(pos)[1]-goals[stg][1]))/8
        pq=[(h(start,0),0,start,si,0,[])]; seen={}; t0=time.time(); res=None
        while pq:
            f,cost,pos,cur,stg,path=heapq.heappop(pq)
            if seen.get((pos,cur,stg),99)<=cost: continue
            seen[(pos,cur,stg)]=cost
            if stg==nst+1: res=(cost,path,pos); break
            if cost>=maxcost or time.time()-t0>cap: continue
            for i in range(n):
                c1=cost+(0 if i==cur else 1)+1
                for c in cand:
                    if c==pos[i] or any(j!=i and abs(c[0]-pos[j][0])<=2 and abs(c[1]-pos[j][1])<=2 for j in range(n)): continue
                    np_=list(pos); np_[i]=c; np_=tuple(np_); cc=centroid(np_)
                    if not mk_ok[cc] or any(ov[o][cc] for o in others): continue
                    ns=stg
                    if stg<nst:
                        A=colors[stg]; later=colors[stg+1:]
                        if any(ov[l][cc] for l in later): continue
                        if stg==0:
                            if full[A](cc): ns=1
                            elif ov[A][cc]: continue
                        else:
                            if ov[A][cc] and not full[A](cc): ns=stg+1
                            elif full[A](cc): continue
                    else:
                        if win[cc]: ns=stg+1
                    if seen.get((np_,i,ns),99)<=c1: continue
                    heapq.heappush(pq,(c1+h(np_,min(ns,nst)),c1,np_,i,ns,path+([('sel',i)] if i!=cur else [])+[('move',i,c)]))
        print(f'  组{gkey}→{rk} colors={colors} plan={res[0] if res else None} ({time.time()-t0:.0f}s)',flush=True)
        if not res: return None,allclicks,ob
        for a in res[1]:
            pt=center(st[a[1]]) if a[0]=='sel' else a[2]; ob=cclick(gm,*pt); allclicks.append(pt)
        print('    标记色',rcols(mk),'levels',ob.levels_completed,'state',ob.state.name,'步',gm._action_count,flush=True)
    return True,allclicks,ob
if __name__=='__main__':
    lvl=int(sys.argv[1]) if len(sys.argv)>1 else 5
    gm,ob=upto(lvl); print('start levels',ob.levels_completed)
    while True:
        cur=ob.levels_completed+1; print(f'== L{cur}')
        ok,clicks,ob=solve_level(gm,ob); print(f'  CLICKS L{cur}:',clicks,flush=True)
        if not ok or ob.levels_completed<cur: print('  ✗ 未过'); break
        if ob.state.name=='WIN': print('WIN 总步数',sum(1 for _ in [])); break
