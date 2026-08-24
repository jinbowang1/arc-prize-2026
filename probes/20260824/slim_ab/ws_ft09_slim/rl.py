import urllib.request, json
g = json.load(urllib.request.urlopen('http://127.0.0.1:19103/grid'))['grid']
# auto: bg = most common color
from collections import Counter
cnt=Counter(v for row in g for v in row)
bg=cnt.most_common(1)[0][0]
s={bg:'.'}
pal={0:'o',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'#',9:'@',10:'A',11:'!',12:'=',13:'B',14:'C',15:'D'}
for r in range(64):
    print('%02d %s'%(r,''.join(pal.get(g[r][c],'?') for c in range(64))))
