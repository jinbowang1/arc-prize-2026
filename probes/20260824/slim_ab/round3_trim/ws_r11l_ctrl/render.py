import json,sys
def load(fn):
    j=json.load(open(fn))
    return j['grid'] if isinstance(j,dict) and 'grid' in j else j
g=load(sys.argv[1])
y0,y1,x0,x1=[int(a) for a in sys.argv[2:6]]
rev={0:'0',1:'1',2:'#',3:'3',5:'.',6:'6',15:'f'}
for r in range(y0,y1+1):
    print(f'{r:2} '+''.join(rev.get(g[r][c],'?') for c in range(x0,x1+1)))
