import json,subprocess,sys
def grid():
    out=subprocess.run(['curl','-s','http://127.0.0.1:19408/grid'],capture_output=True,text=True).stdout
    return json.loads(out)['grid']
def place(g,r0,c0,pat):
    # pat is list of strings, reads 5x5
    for dr,row in enumerate(pat):
        for dc,ch in enumerate(row):
            if ch!='.': assert g[r0+dr][c0+dc]==int(ch,16),(r0+dr,c0+dc,g[r0+dr][c0+dc],ch)
STATIONS=[('T',19,37),('L',34,5),('M',45,15),('B',57,25)]
# but station centers: T center around (21,39); L (36,7); M (47,17); B (59,27)
g=grid()
for name,r,c in [('T',21,39),('L',36,7),('M',47,17),('B',59,27)]:
    print(name, 'center', g[r][c], 'surround', ''.join(format(g[r+d][c+dc],'x') for d in (-1,0,1) for dc in (-2,-1,0,1,2)))
