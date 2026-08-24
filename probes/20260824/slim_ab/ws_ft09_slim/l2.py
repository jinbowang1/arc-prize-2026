import urllib.request, json
g = json.load(urllib.request.urlopen('http://127.0.0.1:19103/grid'))['grid']
col_starts=[12,20,28,36,44]
row_starts=[4,12,20,28,36,44,52]
def content(r0,c0):
    vals=[]
    for dr in range(6):
        for dc in range(6):
            vals.append(g[r0+dr][c0+dc])
    return vals
def classify(vals):
    s=set(vals)
    if s=={8}: return'SOLID8'
    if s=={4}: return'EMPTY4'
    return 'BP:'+','.join(str(x) for x in sorted(s))
for r0 in row_starts:
    for c0 in col_starts:
        vals=content(r0,c0)
        cl=classify(vals)
        if cl=='EMPTY4' and False: pass
        print("r%d c%d -> %s"%(r0,c0,cl))
        if cl.startswith('BP'):
            for dr in range(6):
                print('    '+''.join({0:'o',4:'.',8:'#',12:'=',2:'2'}.get(g[r0+dr][c0+dc],'?') for dc in range(6)))
print("--- top-right marker r0-7 c60-63 ---")
for r in range(0,8):
    print(' r%d '%r + ''.join({0:'o',4:'.',8:'#',12:'=',2:'2'}.get(g[r][c],'?') for c in range(58,64)))
