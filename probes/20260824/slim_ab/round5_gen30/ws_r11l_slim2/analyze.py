import json,subprocess

def getgrid():
    subprocess.run('curl -s http://127.0.0.1:19505/grid -o grid.json',shell=True,check=True)
    return json.load(open('grid.json'))['grid']

def comps(g):
    H=len(g);W=len(g[0])
    seen=[[False]*W for _ in range(H)]
    res=[]
    for r in range(H):
        for c in range(W):
            if g[r][c]==5 or seen[r][c]:continue
            col=g[r][c]
            stack=[(r,c)];seen[r][c]=True;cells=[]
            while stack:
                x,y=stack.pop();cells.append((x,y))
                for dx,dy in((1,0),(-1,0),(0,1),(0,-1)):
                    nx,ny=x+dx,y+dy
                    if 0<=nx<H and 0<=ny<W and not seen[nx][ny] and g[nx][ny]==col:
                        seen[nx][ny]=True;stack.append((nx,ny))
            xs=[a for a,_ in cells];ys=[b for _,b in cells]
            res.append((col,len(cells),(min(xs),max(xs)),(min(ys),max(ys))))
    return res

g=getgrid()
for col,n,(r0,r1),(c0,c1) in comps(g):
    print('color',col,'n',n,'rows',r0,'-',r1,'cols',c0,'-',c1)
