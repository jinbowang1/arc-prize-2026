import urllib.request, json
g = json.load(urllib.request.urlopen('http://127.0.0.1:19103/grid'))['grid']
symb={5:'.',9:'@',8:'#',4:'4',2:'2',0:'o',12:'='}

# Answer region: 5x5 cells of 6x6 each, rows32-61 cols32-61
print("=== ANSWER region 5x5 cells (template chars) ===")
for cr in range(5):
    for cc in range(5):
        r0=32+cr*6; c0=32+cc*6
        cell=''
        for dr in range(6):
            for dc in range(6):
                cell+=symb.get(g[r0+dr][c0+dc],'?')
        print("cell[%d][%d] rows%d-%d cols%d-%d:"%(cr,cc,r0,r0+5,c0,c0+5))
        for dr in range(6):
            r0r=r0+dr
            print('   '+''.join(symb.get(g[r0r][c],'?') for c in range(c0,c0+6)))
