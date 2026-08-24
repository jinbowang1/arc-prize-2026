import json, subprocess, time
def grid():
    return json.loads(subprocess.check_output(["curl","-s","http://127.0.0.1:19504/grid"]))['grid']
def act(x,y):
    return json.loads(subprocess.check_output(["curl","-s","http://127.0.0.1:19504/act?a=6&x=%d&y=%d"%(x,y)]))
for i in range(12):
    g=grid()
    gem=[(r,c) for r in range(64) for c in range(64) if g[r][c]==6]
    if not gem:
        print("no gem found"); break
    gr,gc=gem[0]
    # find color3 cells
    c3=[(r,c) for r in range(64) for c in range(64) if g[r][c]==3]
    in3 = (gr,gc) in c3
    print("step",i,"gem at",(gr,gc),"inside color3?",in3, "color3cells",len(c3))
    if in3:
        print("gem reached color3 block!")
    resp=act(gc,gr)
    print("  done:",resp.get("done"),"effect first:",resp.get("effect","")[:80])
    if resp.get("done") or resp.get("level")!=0 or "level_up" in str(resp.get("effect","")):
        print("WIN detected:",resp); break
    time.sleep(0.05)
