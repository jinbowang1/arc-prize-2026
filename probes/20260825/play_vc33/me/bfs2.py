"""vc33 逐关 BFS v2: 动作=9色按钮 + 1色塞子(闸门); 每个动作后用背景点击冲掉动画; 状态=去掉步数条的画面."""
import sys,json,hashlib,time,collections; sys.path.insert(0,'probes/20260825/play_vc33/me')
from gp import *
LIMIT=int(sys.argv[1]) if len(sys.argv)>1 else 60000
env,o=load('vc33')
P='probes/20260825/play_vc33/me/solutions.json'; sol=json.load(open(P))
for k in sorted(sol, key=lambda s:int(s[1:])):
    for a,x,y in sol[k]: o=act(env,6,x,y)
print('从 level',o.levels_completed,'开始')
def cg(gm):
    c=gm._clean_levels; gm._clean_levels=None; g2=copy.deepcopy(gm); gm._clean_levels=c; g2._clean_levels=c; return g2
def key(g): return hashlib.md5(g[1:].tobytes()).hexdigest()
def do(gm,y,x,lvl):
    """执行动作并冲动画; 返回 (r, frame, 额外动作数)"""
    r=gm.perform_action(ActionInput(id=ACTS[6],data={'x':x,'y':y}),raw=True)
    extra=0; prev=np.array(r.frame[-1]) if r.frame else None
    while r.frame and r.levels_completed==lvl and r.state.name=='NOT_FINISHED' and extra<40:
        r2=gm.perform_action(ActionInput(id=ACTS[6],data={'x':1,'y':10}),raw=True)  # 背景点击
        if not r2.frame: break
        g2=np.array(r2.frame[-1])
        if key(g2)==key(prev) and r2.levels_completed==lvl: 
            # 无变化: 说明不在动画里; 但这一次背景点击已经花了一步 → 回退不可能, 所以只在有动画时才继续
            # 判据: 上一步与本步画面相同且不是动画 → 停; 为避免白花一步, 先用克隆体试探
            break
        r=r2; prev=g2; extra+=1
    return r,prev,extra
def probe_anim(gm,y,x,lvl):
    """在克隆体上判断该动作是否触发动画(下一帧无变化但再点会变)"""
    t=cg(gm); r=t.perform_action(ActionInput(id=ACTS[6],data={'x':x,'y':y}),raw=True)
    if not r.frame: return None
    g1=np.array(r.frame[-1]); t2=cg(t); r2=t2.perform_action(ActionInput(id=ACTS[6],data={'x':1,'y':10}),raw=True)
    if not r2.frame: return None
    g2=np.array(r2.frame[-1]); return key(g2)!=key(g1)
while o.state.name!='WIN':
    lvl=o.levels_completed; g0=grid(o); cs,bg=comps(g0)
    acts=[c['center'] for c in cs if c['color']==9]+[c['center'] for c in cs if c['color']==1]
    print(f'L{lvl+1}: 动作 {len(acts)} 个')
    q=collections.deque([(clone(env),[],g0)]); seen={key(g0)}; found=None; n=0; t0=time.time()
    while q and found is None and n<LIMIT:
        gm,seq,g=q.popleft()
        for (y,x) in acts:
            g2=cg(gm); r=g2.perform_action(ActionInput(id=ACTS[6],data={'x':x,'y':y}),raw=True); n+=1
            steps=[(y,x)]
            if r.levels_completed>lvl or r.state.name=='WIN': found=seq+steps; break
            if not r.frame or r.state.name!='NOT_FINISHED': continue
            g1=np.array(r.frame[-1])
            if key(g1)==key(g):   # 画面没变: 可能是动画开始; 用背景点击冲
                for i in range(40):
                    r=g2.perform_action(ActionInput(id=ACTS[6],data={'x':1,'y':10}),raw=True); steps.append((10,1))
                    if r.levels_completed>lvl or r.state.name=='WIN': found=seq+steps; break
                    if not r.frame or r.state.name!='NOT_FINISHED': break
                    gn=np.array(r.frame[-1])
                    if key(gn)==key(g1) and i>0: break
                    g1=gn
                if found: break
                if not r.frame or r.state.name!='NOT_FINISHED' or key(g1)==key(g): continue
            h=key(g1)
            if h in seen: continue
            seen.add(h); q.append((g2,seq+steps,g1))
    if not found: print(f'L{lvl+1} 未解出 节点{n} 状态{len(seen)} {time.time()-t0:.0f}s'); break
    print(f'L{lvl+1} 解出 {len(found)} 步 (节点{n} 状态{len(seen)} {time.time()-t0:.0f}s)')
    seq=[[6,x,y] for y,x in found]
    for a,x,y in seq: o=act(env,6,x,y)
    print('  真机 level',o.levels_completed,'state',o.state.name)
    if o.levels_completed<=lvl: print('  真机没过关?!'); break
    sol[f'L{lvl+1}']=seq; json.dump(sol,open(P,'w'))
print('结束',o.state.name,{k:len(v) for k,v in sol.items()},'人类 [7,18,44,61,131,34,152]')
