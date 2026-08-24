import json, sys, collections

def load(fn):
    j=json.load(open(fn))
    g=j['grid'] if 'grid' in j else j
    if isinstance(g,dict):
        for k,v in g.items():
            if isinstance(v,list) and len(v)==64: g=v
    return g

a=load(sys.argv[1]); b=load(sys.argv[2])
A=[(r,c) for r in range(64) for c in range(1,64) if a[r][c]!=5]
B=[(r,c) for r in range(64) for c in range(1,64) if b[r][c]!=5]
def bycolor(cells,g):
    d={}
    for (r,c) in cells: d.setdefault(g[r][c],[]).append((r,c))
    return d
da=bycolor(A,a); db=bycolor(B,b)
for color in sorted(set(da)|set(db)):
    ca=set(da.get(color,[])); cb=set(db.get(color,[]))
    gone=sorted(ca-cb); added=sorted(cb-ca); 
    print(f"色{color}: -{len(gone)}格 +{len(added)}格")
    if gone: print(f"  消失: {gone}")
    if added: print(f"  新增: {added}")
