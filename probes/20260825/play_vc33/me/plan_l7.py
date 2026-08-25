import sys,json; sys.path.insert(0,'probes/20260825/play_vc33/me')
from gp import *
env,o=load('vc33')
P='probes/20260825/play_vc33/me/solutions.json'; sol=json.load(open(P))
for k in ['L1','L2','L3','L4','L5','L6']:
    for a,x,y in sol[k]: o=act(env,6,x,y)
g=clone(env); seq=[]; cur=grid(o)
def levels(gr):
    L={}
    for name,(c0,c1,r0,r1) in {'上左':(8,21,8,29),'中':(26,39,8,55),'上右':(44,55,8,29),'下左':(8,21,32,55),'下右':(44,55,32,55)}.items():
        L[name]=int((gr[r0:r1+1,c0]==0).sum())   # 只沿舱左壁一列数, 避开船
    return L
def boats(gr):
    d={}
    for col,n in [(11,'b'),(14,'e'),(15,'f')]:
        cs,_=comps(np.where(gr==col,col,3),bg=3); cs=sorted(cs,key=lambda c:c['n'])
        if cs: d[n]=cs[-1]['center']
    return d
BTN={('上左','中'):(8,20),('中','上左'):(8,24),('中','上右'):(8,38),('上右','中'):(8,42),('下左','中'):(32,20),('中','下左'):(32,24),('中','下右'):(32,38),('下右','中'):(32,42)}
# BTN[(dst,src)] = 把 2 行水从 src 移到 dst 的按钮  (按探测: (8,20): 上左+2 中-2 → dst 上左 src 中)
def press(y,x,tag=''):
    global cur
    r=g.perform_action(ActionInput(id=ACTS[6],data={'x':x,'y':y}),raw=True); seq.append((y,x))
    if not r.frame: print('空帧!',tag); return r
    new=np.array(r.frame[-1]); changed=(new[1:]!=cur[1:]).sum()>0; cur=new
    if not changed: print('  ⚠️ 无变化',tag,(y,x))
    return r
def move(src,dst,rows):
    for _ in range(rows//2): press(*BTN[(dst,src)],tag=f'{src}->{dst}')
def set_levels(targets):
    # 先把高于目标的舱放水进中舱, 再从中舱给低于目标的舱补水
    L=levels(cur)
    for ch,t in targets.items():
        if ch!='中' and L[ch]>t: move(ch,'中',L[ch]-t)
    L=levels(cur)
    for ch,t in targets.items():
        if ch!='中' and L[ch]<t: move('中',ch,t-L[ch])
    L=levels(cur)
    if '中' in targets and L['中']!=targets['中']:
        d=L['中']-targets['中']; spill=[c for c in ('上左','下右','上右','下左') if c not in targets]
        for c in spill:
            if d==0: break
            cap={'上左':22,'上右':22,'下左':24,'下右':24}[c]
            if d>0: k=min(d,cap-L[c]); move('中',c,k); d-=k
            else: k=min(-d,L[c]); move(c,'中',k); d+=k
    print('   水位',levels(cur),'船',boats(cur))
def plug(y,x,tag):
    r=press(y,x,tag); print(f'   闸门{tag}: frames={len(r.frame)} 船={boats(cur)} level={r.levels_completed}'); return r
print('初始',levels(cur),boats(cur))
print('P1 f 上右→中'); set_levels({'中':8,'上右':8}); plug(19,40,'P_UR')
print('P2 f 中→下左 (b 换进中)'); set_levels({'中':30,'下左':6}); plug(41,22,'P_DL')
print('P3 b 中→上右'); set_levels({'中':8,'上右':8}); plug(19,40,'P_UR')
print('P4 e 下右→中'); set_levels({'中':30,'下右':6}); r=plug(41,40,'P_DR')
print('P5 终态'); set_levels({'中':10,'上右':18,'下左':18})
r=press(10,1,'收尾'); print('最终',levels(cur),boats(cur),'level',r.levels_completed,r.state.name,'步数',len(seq))
if r.levels_completed>6 or r.state.name=='WIN':
    s=[[6,x,y] for y,x in seq]; sol['L7']=s; json.dump(sol,open(P,'w')); print('已存')
