import json, sys, collections

def load(fn):
    j=json.load(open(fn))
    grid=j['grid'] if 'grid' in j else j
    if isinstance(grid,dict):
        for k,v in grid.items():
            if isinstance(v,list) and len(v)==64: grid=v
    return grid

def blobs(grid, colors=None, exclude_col0=True):
    H=len(grid);W=len(grid[0])
    seen=set()
    res=[]
    for r in range(H):
        for c in range(W):
            if exclude_col0 and c==0: continue
            col=grid[r][c]
            if col==5: continue
            if colors and col not in colors: continue
            if (r,c) in seen: continue
            # BFS
            stack=[(r,c)]; seen.add((r,c)); cells=[]
            while stack:
                cr,cc=stack.pop(); cells.append((cr,cc))
                for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
                    nr,nc=cr+dr,cc+dc
                    if 0<=nr<H and 0<=nc<W and (nr,nc) not in seen and grid[nr][nc]==col:
                        seen.add((nr,nc)); stack.append((nr,nc))
            res.append((col,cells))
    return res

if __name__=='__main__':
    fn=sys.argv[1]
    grid=load(fn)
    col=sys.argv[2] if len(sys.argv)>2 else None
    colors=None if col is None else set(int(x) for x in col.split(','))
    for bcol,cells in blobs(grid,colors):
        rs=[c[0] for c in cells]; cs=[c[1] for c in cells]
        print(f"色{bcol} {len(cells)}格 r{min(rs)}-{max(rs)} c{min(cs)}-{max(cs)}  n={len(cells)}")
        # print cells sorted
        print("   ",sorted(cells))
