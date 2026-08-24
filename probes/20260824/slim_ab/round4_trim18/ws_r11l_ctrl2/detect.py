import json,subprocess
def grid():
    return json.loads(subprocess.run(['curl','-s','http://127.0.0.1:19408/grid'],capture_output=True,text=True).stdout)['grid']
g=grid()
# Print all cells that are not 5 and not 2 and not 0-border, grouped, with coords
cells={}
for r in range(1,64):
    for c in range(1,64):
        v=g[r][c]
        if v not in (5,2,0):
            cells.setdefault(v,[]).append((r,c))
for v in sorted(cells):
    pts=cells[v]
    # find connected comps
    comps=[]
    seen=set()
    from collections import deque
    for p in pts:
        if p in seen: continue
        q=deque([p]);seen.add(p);comp=[]
        while q:
            rr,cc=q.popleft();comp.append((rr,cc))
            for dr,dc in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
                np=(rr+dr,cc+dc)
                if np in pts and np not in seen:
                    seen.add(np);q.append(np)
        comps.append(comp)
    print(f'color {v}: {len(pts)} cells in {len(comps)} comp')
    for comp in sorted(comps,key=lambda c:(min(x for x,y in c),min(y for x,y in c))):
        rmin=min(x for x,y in comp);rmax=max(x for x,y in comp);cmin=min(y for x,y in comp);cmax=max(y for x,y in comp)
        if len(comp)>=4:
            art=[]
            for rr in range(rmin,rmax+1):
                art.append(''.join(format(g[rr][cc],'x') if (rr,cc) in set(comp) else '.' for cc in range(cmin,cmax+1)))
            print('   ',len(comp),'cells box',rmin,rmax,cmin,cmax)
            for line in art: print('      ',line)
