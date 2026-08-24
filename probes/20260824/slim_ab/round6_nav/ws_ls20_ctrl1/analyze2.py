import json
from collections import deque
g=json.load(open('/tmp/g.json'))['grid']
CH=64//5; CW=64//5
openb=[[False]*CW for _ in range(CH)]
for cy in range(CH):
    for cx in range(CW):
        openb[cy][cx]=(set(g[cy*5+dy][cx*5+dx] for dy in range(5) for dx in range(5))=={3})
pcx, pcy = 6,9
seen=set(); q=deque([(pcx,pcy)])
while q:
    cx,cy=q.popleft()
    if not openb[cy][cx] or (cx,cy) in seen: continue
    seen.add((cx,cy))
    for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
        nx,ny=cx+dx,cy+dy
        if 0<=nx<CW and 0<=ny<CH and (nx,ny) not in seen and openb[ny][nx]:
            q.append((nx,ny))
for cy in range(CH):
    row=''
    for cx in range(CW):
        if (cx,cy)==(pcx,pcy): row+='P'
        elif openb[cy][cx]: row+=('@' if (cx,cy) in seen else '.')
        else: row+='#'
    print('%02d %s'%(cy*5,row))
print('reachable count:',len(seen))