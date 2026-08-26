import re,glob,sys,statistics,collections
def sect(b):
    idx=[(m.start(),m.group().strip()) for m in re.finditer(r'\n\[[A-Z][A-Z _a-z:]*[^\n]*\]\n',b)]; idx.append((len(b),''))
    return {idx[i][1]:b[idx[i][0]:idx[i+1][0]] for i in range(len(idx)-1)}
def run_metrics(path):
    t=open(path).read()
    parts=re.split(r'\n--- analysis_step=(\d+) \| action=(\d+) \| (\d\d:\d\d:\d\d) \| ([\w-]+) ---\n',t)
    turns=[(int(parts[i]),int(parts[i+1]),parts[i+2],parts[i+4]) for i in range(1,len(parts),5)]
    n=len(turns); idle=0; clicks=[]; keys=0; lc=0; go=0; first_lc=None; actions=0
    for i,(a,act,ts,b) in enumerate(turns):
        s=sect(b); tr=s.get('[TOOL RESULT: python]',''); tc=s.get('[TOOL CALL: python]','')
        nxt=turns[i+1][1] if i+1<len(turns) else None
        if nxt is not None and nxt==act: idle+=1
        for m in re.finditer(r"'row':\s*(\d+),\s*'col':\s*(\d+)",tc): clicks.append((int(m.group(1))//4,int(m.group(2))//4))
        if 'level_completed: true' in tr or "'level_completed': True" in tr:
            lc+=1; first_lc=first_lc or act
        if 'game_over: true' in tr or "'game_over': True" in tr: go+=1
    final=re.findall(r'Current state: step (\d+), level (\d+)',t)
    last_step=int(final[-1][0]) if final else 0; last_level=int(final[-1][1]) if final else 1
    breadth30=len(set(clicks[:30])); breadth_all=len(set(clicks))
    return dict(turns=n,idle=idle,idle_pct=round(100*idle/max(n,1)),steps=last_step,level=last_level,lc=lc,deaths=go,first_lc=first_lc,breadth30=breadth30,breadth=breadth_all,clicks=len(clicks))
def summarize(label,files):
    rows=[(f,run_metrics(f)) for f in files]
    keys=['turns','idle','idle_pct','steps','lc','deaths','breadth30','breadth']
    print(f"== {label} n={len(rows)}: "+"  ".join(f"{k}={statistics.mean(r[k] for _,r in rows):.1f}" for k in keys))
    return rows
if __name__=='__main__':
    for label,pat in [('kaggle official','kaggle_tx_official/transcripts/*.txt'),('kaggle handbook','kaggle_tx_handbook/transcripts/*.txt'),('q27 ctrl','duck_q27_ab/*ab-ctrl/transcripts/*.txt'),('q27 guide','duck_q27_ab/*ab-guide/transcripts/*.txt')]:
        fs=sorted(glob.glob(pat))
        if fs:
            rows=summarize(label,fs)
            if len(sys.argv)>1:
                for f,r in rows: print('   ',f.split('/')[-1][:16],r)
