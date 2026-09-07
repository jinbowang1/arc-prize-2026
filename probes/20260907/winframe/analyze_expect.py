"""统计模型有没有用 expect, 用得准不准。

三个指标(按 notes/plan-20260907-expect-lever.md 定的):
1. 模型填不填     -> 带 expect 的 action 调用占比
2. 填得准不准     -> prediction_mismatch 触发次数
3. 换来过关了吗   -> 过关数(另由 analyze_ab 给)
"""
import json, re, sys, glob, os

run_dir = sys.argv[1] if len(sys.argv) > 1 else "."
codes, calls_with_expect, calls_total = [], 0, 0

for f in (glob.glob(os.path.join(run_dir, "*_requests.jsonl")) or glob.glob(os.path.join(run_dir, "requests.jsonl"))):
    for line in open(f, encoding="utf-8"):
        try: d = json.loads(line)
        except Exception: continue
        for m in d.get("messages") or []:
            for tc in m.get("tool_calls") or []:
                args = (tc.get("function") or {}).get("arguments") or ""
                if isinstance(args, str):
                    try: a = json.loads(args)
                    except Exception: continue
                else: a = args
                c = a.get("code")
                if c: codes.append(c)

uniq = list(dict.fromkeys(codes))
for c in uniq:
    n = len(re.findall(r"\baction\s*\(", c))
    calls_total += n
    if "expect" in c:
        calls_with_expect += len(re.findall(r"['\"]expect['\"]", c))

print(f"去重代码段: {len(uniq)}")
print(f"含 'expect' 的代码段: {sum(1 for c in uniq if 'expect' in c)}/{len(uniq)}")
print(f"action() 调用总数(去重后): {calls_total}")
print(f"其中带 expect 的动作对象: {calls_with_expect}")

# 失配触发
mm = 0
for f in (glob.glob(os.path.join(run_dir, "*_requests.jsonl")) or glob.glob(os.path.join(run_dir, "requests.jsonl"))):
    s = open(f, encoding="utf-8").read()
    mm += s.count("prediction_mismatch")
print(f"prediction_mismatch 出现: {mm}")

# 样例
for c in uniq:
    if "expect" in c:
        print("\n--- 模型用 expect 的样例 ---")
        print(c[:600]); break
else:
    print("\n(模型一次都没用 expect)")
