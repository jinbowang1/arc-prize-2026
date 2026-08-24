import json, urllib.request, subprocess, time
def g():
    return json.load(urllib.request.urlopen('http://127.0.0.1:19528/grid'))['grid']
def info():
    gr=g()
    for cc in [0,5,4]:
        cells=[(r,c) for r in range(64) for c in range(64) if gr[r][c]==cc]
        if cells:
            rs=[r for r,c in cells];cs=[c for r,c in cells]
            print(f'   c{cc}: n{len(cells)} y{min(rs)}-{max(rs)} x{min(cs)}-{max(cs)}')
        else:
            print(f'   c{cc}: 0')
    dots=[(c,r) for r in range(64) for c in range(64) if gr[r][c]==0]
    print('   dots:',dots)
info()
for i in range(1,7):
    subprocess.run(['curl','-s','http://127.0.0.1:19528/act?a=4'],capture_output=True)
    print(f'--- after a=4 x{i} ---')
    info()
