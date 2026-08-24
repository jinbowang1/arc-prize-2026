import json, urllib.request, sys
def getgrid():
    return json.load(urllib.request.urlopen('http://127.0.0.1:19528/grid'))['grid']
def diffsets(b,a):
    ch=[(x,y,b[y][x],a[y][x]) for y in range(64) for x in range(64) if b[y][x]!=a[y][x]]
    return ch
def show(ch):
    for x,y,f,t in sorted(ch):
        print(f'({x},{y}) {f}->{t}',end='  ')
    print(f'  [n={len(ch)}]')
import subprocess
# restore dots position? just record current
b=getgrid()
key=sys.argv[1]
subprocess.run(['curl','-s',f'http://127.0.0.1:19528/act?a={key}'],capture_output=True)
a=getgrid()
show(diffsets(b,a))
# also print 0-dot positions and color ranges
for cc in [0,5,4]:
    cells=[(r,c) for r in range(64) for c in range(64) if a[r][c]==cc]
    if cells:
        rs=[r for r,c in cells];cs=[c for r,c in cells]
        print(f'color{cc}: n{len(cells)} rows{min(rs)}-{max(rs)} cols{min(cs)}-{max(cs)}')
