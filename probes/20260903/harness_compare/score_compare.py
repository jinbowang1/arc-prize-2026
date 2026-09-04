#!/usr/bin/env python3
"""两套 harness 同题对照的统计脚本。两边用同一份代码、同一套判据。

判据(写死在这里, 报告里逐条抄):
  levels_completed  = taaf 自己记的过关数(benchmark.json 里的 levels_completed)
  final_score       = taaf 自己算的分(每关 (基准步/实际步)^2 封顶1.15 按关序加权)
  actions           = 这一路总共对环境下了多少个动作(= history 长度)
  board_changed_rate= events.jsonl 里 type=="action" 且 board_changed 为真的比例
                      注意: HUD/倒计时变化也算 board_changed, 这是上限口径, 不是"有用动作"
  llm_calls         = requests.jsonl 里 event=="request" 的条数(= 发给模型多少次)
  in_chars_per_call = 每次请求 messages 拼起来的字符数均值(hub 不回 prompt_tokens 明细, 用字符数代)
  gen_tokens        = benchmark.json history 里 generated_tokens 之和
  wallclock         = final_wallclock_seconds
"""
import json, glob, os, sys, statistics, collections

HC = os.path.dirname(os.path.abspath(__file__))

def msg_chars(messages):
    n = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str): n += len(c)
        elif isinstance(c, list): n += len(json.dumps(c, ensure_ascii=False))
    return n

def load_side(side, run_glob):
    dirs = sorted(glob.glob(os.path.join(HC, f"{side}_runs", run_glob)))
    rows = []
    for d in dirs:
        bj = os.path.join(d, "benchmark.json")
        if not os.path.exists(bj): continue
        bm = json.load(open(bj))
        # 每个 game_run 是一路(一局一个 pass)
        per_game_pass = collections.defaultdict(int)
        for gr in bm["game_runs"]:
            gid = gr["game_id"]; short = gid.split("-")[0]
            p = per_game_pass[gid]; per_game_pass[gid] += 1
            hist = gr.get("history") or []
            gen = sum(h.get("generated_tokens") or 0 for h in hist)
            # events
            ev = os.path.join(d, "artifacts", f"{gid}_p{p}_events.jsonl")
            acts = 0; changed = 0; batched = 0
            if os.path.exists(ev):
                for line in open(ev):
                    e = json.loads(line)
                    if e.get("type") == "action":
                        acts += 1
                        if e.get("board_changed"): changed += 1
                        if (e.get("batch_size") or 1) > 1: batched += 1
            # requests: 主跑是每局一份 <gid>_p<n>_requests.jsonl, 冒烟是整 run 一份 requests.jsonl
            calls = 0; chars_tot = 0; resp = 0
            rq_game = os.path.join(d, f"{gid}_p{p}_requests.jsonl")
            if os.path.exists(rq_game):
                for line in open(rq_game):
                    try: e = json.loads(line)
                    except Exception: continue
                    if e.get("event") == "request":
                        calls += 1; chars_tot += msg_chars(e.get("messages") or [])
                    elif e.get("event") == "response":
                        resp += 1
            rows.append(dict(llm_calls=calls,
                in_chars_avg=(chars_tot / calls if calls else 0),
                responses=resp, lost_calls=(calls - resp),side=side, run=os.path.basename(d), game=short, gid=gid, p=p,
                levels=gr.get("levels_completed", 0), score=gr.get("final_score", 0.0),
                state=gr.get("state"), actions=len(hist), env_actions=acts,
                changed=changed, batched=batched,
                apl=gr.get("actions_per_level"), base=gr.get("base_actions_per_level"),
                gen_tokens=gen, wall=gr.get("final_wallclock_seconds") or 0.0))
        # requests.jsonl 是整个 run 一份, 按 run 聚合 LLM 调用
        rq = os.path.join(d, "requests.jsonl")
        if os.path.exists(rq):
            n = 0; tot = 0
            for line in open(rq):
                try: e = json.loads(line)
                except Exception: continue
                if e.get("event") != "request": continue
                n += 1; tot += msg_chars(e.get("messages") or [])
            for r in rows:
                if r["run"] == os.path.basename(d):
                    r["run_llm_calls"] = n
                    r["run_in_chars_avg"] = tot / n if n else 0
    return rows

def main():
    pat = sys.argv[1] if len(sys.argv) > 1 else "*main-*"
    allrows = load_side("duck", pat.replace("*main-*", "*main-duck")) + \
              load_side("son",  pat.replace("*main-*", "*main-son"))
    if not allrows:
        print("NO ROWS — 没有任何 run 目录匹配", pat); return
    json.dump(allrows, open(os.path.join(HC, "rows.json"), "w"), indent=1, ensure_ascii=False)

    # 跑到一半的路: final_score 还是 None。单独统计, 不混进对照
    incomplete = [r for r in allrows if r.get("score") is None or r.get("state") is None]
    for r in incomplete:
        r["score"] = r.get("score") or 0.0
        r["levels"] = r.get("levels") or 0
    if incomplete:
        by = collections.Counter((r["side"], r["game"]) for r in incomplete)
        print(f"⚠️ 未跑完的路 {len(incomplete)} 条(已从对照里剔除): {dict(by)}\n")
    allrows = [r for r in allrows if r not in incomplete]
    if not allrows:
        print("没有跑完的路可比"); return
    games = sorted({r["game"] for r in allrows})
    print(f"{'局':<6}{'套':<6}{'n':<3}{'过关(逐种子)':<16}{'均过关':<8}{'分(逐种子)':<22}{'均分':<8}"
          f"{'动作':<8}{'变盘率':<8}{'生成token':<11}{'用时s':<7}")
    print("-"*112)
    agg = {}
    for g in games:
        for side in ("duck", "son"):
            rs = [r for r in allrows if r["game"] == g and r["side"] == side]
            if not rs: continue
            lv = [r["levels"] for r in rs]; sc = [round(r["score"], 2) for r in rs]
            act = [r["env_actions"] for r in rs]
            ch = sum(r["changed"] for r in rs); ac = sum(r["env_actions"] for r in rs)
            gt = [r["gen_tokens"] for r in rs]; wl = [r["wall"] for r in rs]
            agg[(g, side)] = dict(n=len(rs), lv=lv, sc=sc,
                mean_lv=statistics.mean(lv), mean_sc=statistics.mean([r["score"] for r in rs]),
                mean_act=statistics.mean(act), chrate=(ch/ac if ac else 0),
                mean_gt=statistics.mean(gt), mean_wl=statistics.mean(wl))
            a = agg[(g, side)]
            print(f"{g:<6}{side:<6}{a['n']:<3}{str(lv):<16}{a['mean_lv']:<8.2f}{str(sc):<22}"
                  f"{a['mean_sc']:<8.2f}{a['mean_act']:<8.1f}{a['chrate']*100:<8.1f}"
                  f"{a['mean_gt']:<11.0f}{a['mean_wl']:<7.0f}")
        print()
    print("="*112)
    for side in ("duck", "son"):
        rs = [r for r in allrows if r["side"] == side]
        n = len(rs)
        tl = sum(r["levels"] for r in rs); ts = sum(r["score"] for r in rs)
        ta = sum(r["env_actions"] for r in rs); tc = sum(r["changed"] for r in rs)
        tg = sum(r["gen_tokens"] for r in rs); tw = sum(r["wall"] for r in rs)
        calls = {r["run"]: r.get("run_llm_calls", 0) for r in rs}
        chars = {r["run"]: r.get("run_in_chars_avg", 0) for r in rs}
        print(f"[{side}] 路数={n} 总过关={tl} 均分={ts/n:.3f} 总动作={ta} "
              f"变盘率={tc/ta*100 if ta else 0:.1f}% 生成token={tg} 总用时={tw/60:.1f}min "
              f"LLM调用={sum(calls.values())} 平均每次输入字符={statistics.mean([v for v in chars.values() if v]) if any(chars.values()) else 0:.0f}")
        if ta: print(f"        每动作生成token={tg/ta:.0f}  每动作用时={tw/ta:.1f}s  "
                     f"动作/分钟={ta/(tw/60):.2f}")
        pc = sum(r.get("llm_calls", 0) for r in rs)
        lost = sum(r.get("lost_calls", 0) for r in rs)
        ch_avg = [r.get("in_chars_avg", 0) for r in rs if r.get("in_chars_avg")]
        if pc:
            print(f"        逐局请求日志: 调用={pc} 无回包={lost}({lost/pc*100:.1f}%) "
                  f"每次输入字符均值={statistics.mean(ch_avg):.0f}")

if __name__ == "__main__":
    main()
