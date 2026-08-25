"""L7 束搜索: 状态=克隆体, 启发式=c尖端(色12段内的d)到目标(16,25)的曼哈顿距离(+a尖端离(7,22)的距离), 按画面去重."""
import sys,json,time,hashlib; sys.path.insert(0,'probes/20260825/play_s5i5')
from gp import *
env,o=load('s5i5')
sol=json.load(open('probes/20260825/play_s5i5/solutions.json'))
for k in ['L1','L2','L3','L4','L5','L6']:
    for a,x,y in sol[k]: o=act(env,6,x,y)
C={'a-':(51,53),'a+':(51,60),'8r':(58,60),'b-':(51,3),'b+':(51,10),'br':(51,17),'9-':(51,25),'9+':(51,32),'9r':(51,39),'e-':(58,3),'e+':(58,10),'er':(58,17),'c-':(58,25),'c+':(58,32)}
T2=(16,25); T1=(7,22)
def tips(g):
    out={}
    for col,name in [(12,'c'),(10,'a')]:
        ys,xs=np.where(g[:45]==col)
        if len(ys)==0: out[name]=None; continue
        # 尖端 d(13) 与该色相邻的格
        m=np.zeros_like(g,bool); m[ys,xs]=True
        best=None
        for y,x in np.argwhere(g[:45]==13):
            nb=[(y+dy,x+dx) for dy,dx in((1,0),(-1,0),(0,1),(0,-1))]
            if any(0<=a<64 and 0<=b<64 and m[a,b] for a,b in nb): best=(int(y),int(x)); break
        out[name]=best
    return out
def h(g):
    t=tips(g); s=0
    s+= (abs(t['c'][0]-T2[0])+abs(t['c'][1]-T2[1])) if t['c'] else 60
    s+= 0.5*((abs(t['a'][0]-T1[0])+abs(t['a'][1]-T1[1])) if t['a'] else 60)
    return s
def clonegame(gm):
    clean=gm._clean_levels; gm._clean_levels=None; g2=copy.deepcopy(gm); gm._clean_levels=clean; g2._clean_levels=clean; return g2
root=clone(env); g0=grid(o)
beam=[(h(g0),[],root,g0)]; seen={hashlib.md5(g0[:45].tobytes()).hexdigest()}
W=int(sys.argv[1]) if len(sys.argv)>1 else 120; D=int(sys.argv[2]) if len(sys.argv)>2 else 110
t0=time.time()
for depth in range(1,D+1):
    cand=[]
    for hh,seq,gm,g in beam:
        for k,(y,x) in C.items():
            g2=clonegame(gm); r=g2.perform_action(ActionInput(id=ACTS[6],data={'x':x,'y':y}),raw=True); g1=np.array(r.frame[-1])
            if (g1[:45]!=g[:45]).sum()==0: continue
            key=hashlib.md5(g1[:45].tobytes()).hexdigest()
            if key in seen: continue
            seen.add(key)
            if r.levels_completed>6:
                print('SOLVED depth',depth,'seq',seq+[k],flush=True); json.dump(seq+[k],open('probes/20260825/play_s5i5/l7_beam_solution.json','w')); sys.exit(0)
            cand.append((h(g1),seq+[k],g2,g1))
    if not cand: print('dead end at depth',depth); break
    cand.sort(key=lambda c:c[0]); beam=cand[:W]
    print(f'depth {depth} best_h {beam[0][0]:.1f} tips {tips(beam[0][3])} cand {len(cand)} {time.time()-t0:.0f}s seq_head {beam[0][1][-6:]}',flush=True)
print('未解出')
