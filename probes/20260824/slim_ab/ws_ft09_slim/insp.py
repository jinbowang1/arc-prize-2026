import urllib.request, json
g = json.load(urllib.request.urlopen('http://127.0.0.1:19103/grid'))['grid']
s={5:'.',9:'@',8:'#',4:'4',2:'2',0:'o',11:'!',12:'='}
print("answer frame rows32-61 cols32-61:")
for r in range(32,62):
    print(''.join(s.get(g[r][c],'?') for c in range(32,62)))
