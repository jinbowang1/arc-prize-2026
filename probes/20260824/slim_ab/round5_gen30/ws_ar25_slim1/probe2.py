import json,urllib.request,subprocess,sys
def g(): return json.load(urllib.request.urlopen('http://127.0.0.1:19528/grid'))['grid']
def dump():
    gr=g(); print('   '+''.join(str(c%10) for c in range(60)))
    for r in range(10,24):
        print(f'{r:3d} '+''.join('.' if gr[r][c]==9 else chr(48+gr[r][c]) for c in range(60)))
key=sys.argv[1]
print('BEFORE a=%s'%key); dump()
subprocess.run(['curl','-s',f'http://127.0.0.1:19528/act?a={key}'],capture_output=True)
print('AFTER a=%s'%key); dump()
