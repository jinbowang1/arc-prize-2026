import json,urllib.request,subprocess
def g(): return json.load(urllib.request.urlopen('http://127.0.0.1:19528/grid'))['grid']
def col63():
    gr=g()
    return [(r,gr[r][63]) for r in range(64)]
print('col63 start:', col63())
for i in range(3):
    subprocess.run(['curl','-s','http://127.0.0.1:19528/act?a=2'],capture_output=True)
    print(f'after a=2 x{i+1} col63:',col63())
