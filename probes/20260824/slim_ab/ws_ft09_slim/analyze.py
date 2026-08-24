import urllib.request, json
g = json.load(urllib.request.urlopen('http://127.0.0.1:19103/grid'))['grid']
symb = {5:'.',9:'@',8:'#',4:'4',2:'2',0:'o',12:'=','1':'1'}
for r in range(64):
    row=''
    for c in range(64):
        v=g[r][c]
        row+=symb.get(v,str(v))
    print('%02d %s'%(r,row))
