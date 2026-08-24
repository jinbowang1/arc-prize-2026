import json,subprocess
def getgrid():
    return json.loads(subprocess.run(['curl','-s','http://127.0.0.1:19408/grid'],capture_output=True,text=True).stdout)['grid']
def act(x,y):
    return json.loads(subprocess.run(['curl','-s',f'http://127.0.0.1:19408/act?a=6&x={x}&y={y}'],capture_output=True,text=True).stdout)
def occ(g):
    st={'T':(19,37),'L':(34,5),'M':(45,15),'B':(57,25)}
    centers={'T':(21,39),'L':(36,7),'M':(47,17),'B':(59,27)}
    res={}
    for name,(r0,c0) in st.items():
        cr,cc=centers[name]
        cx=g[cr][cc]
        # detect by body color around center
        body=g[cr][cc]
        # count distinct colors in 3x3
        vals=set(g[r][c] for r in range(r0+1,r0+4) for c in range(c0+1,c0+4))
        res[name]={'center':cx,'body3':vals}
    return res
for name,(x,y) in [('B',(27,59)),('T',(39,21)),('L',(7,36)),('M',(17,47))]:
    r=act(x,y)
    print('clicked',name,'steps',r['steps_used'],'changed',r['changed_cells'],'done',r['done'])
    print('  ',occ(getgrid()))
    if r.get('level_up'): print('LEVEL UP!',r)
