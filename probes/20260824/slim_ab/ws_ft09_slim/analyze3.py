import urllib.request, json
g = json.load(urllib.request.urlopen('http://127.0.0.1:19103/grid'))['grid']
symb={5:'.',9:'@',8:'#',4:'4',2:'2',0:'o',12:'='}
def pat(r0,c0):
    return tuple(tuple(g[r0+dr][c0+dc] for dc in range(6)) for dr in range(6))

# enumerate all 6x6 tiles anchored where top-left!=5 and forms block
seen=set()
tiles=[]
for r in range(59):
    for c in range(59):
        if g[r][c]!=5 and (r,c) not in seen:
            # bounding of component
            # gather component
            # simple: trace component
            comp=[(r,c)]
            v=(r)
            # BFS
            q=[(r,c)]
            cc={(r,c)}
            while q:
                x,y=q.pop()
                for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                    nx,ny=x+dx,y+dy
                    if 0<=nx<64 and 0<=ny<64 and g[nx][ny]!=5 and (nx,ny) not in cc:
                        cc.add((nx,ny)); q.append((nx,ny))
            if len(cc)==36:
                rr=[p[0] for p in cc]; cc2=[p[1] for p in cc]
                r0=min(rr);c0=min(cc2)
                if (r0,c0) not in seen:
                    seen.add((r0,c0))
                    tiles.append((r0,c0))
            seen|={(p) for p in cc}
# also need answer region handled separately
print("Number of 6x6 tiles:", len(tiles))
for r0,c0 in sorted(tiles,key=lambda t:(t[0],t[1])):
    print("tile at r%d c%d:"%(r0,c0))
    for dr in range(6):
        print('   '+''.join(symb.get(g[r0+dr][c],'?') for c in range(c0,c0+6)))
