import re,sys,json
def parse(path):
    txt=open(path).read()
    parts=re.split(r'\n--- analysis_step=(\d+) \| action=(\d+) \| (\d\d:\d\d:\d\d) \| ([\w-]+) ---\n',txt)
    turns=[]
    for i in range(1,len(parts),5):
        st,act,ts,kind,body=parts[i],parts[i+1],parts[i+2],parts[i+3],parts[i+4]
        m=re.search(r'Current state: step (\d+), level (\d+)',body)
        turns.append(dict(analysis=int(st),action=int(act),ts=ts,kind=kind,step=int(m.group(1)) if m else None,level=int(m.group(2)) if m else None,body=body))
    return turns
if __name__=='__main__':
    for v in ['official','handbook']:
        t=parse(f'sb26_{v}/transcripts/sb26-7fbdac44_p0.txt')
        print(f'== {v}: {len(t)} turns, first {t[0]["ts"]} last {t[-1]["ts"]}, kinds', {k:sum(1 for x in t if x["kind"]==k) for k in set(x["kind"] for x in t)})
        lv=None
        for x in t:
            if x['level']!=lv:
                print(f'   level {x["level"]} reached at analysis {x["analysis"]} step {x["step"]} {x["ts"]}'); lv=x['level']
        print('   last:',t[-1]['analysis'],t[-1]['step'],t[-1]['level'],t[-1]['ts'])
