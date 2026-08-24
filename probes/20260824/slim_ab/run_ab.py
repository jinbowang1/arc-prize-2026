"""/state 瘦身 A/B: 对照组=现状(frame+summary), 实验组=A3_STATE_SLIM=1(仅summary, 原文靠/grid)。
2 局(r11l/ft09) × 2 组 = 4 路并行, 各自 game_server 进程+独立 DSH_HOME。
判据(用户规矩): 至少 2局×2组, 方向一致才算数; 比过关数, 平局比步数效率。
key 从环境 DEEPSEEK_API_KEY 注入, 不落文件。"""
import json, os, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT = Path(os.path.expanduser("~/Desktop/project/arc-agi-3"))
DSH = Path(os.path.expanduser("~/Desktop/project/deepseek-harness/apps/cli/lib/bin.js"))
PATCH_CTRL = ROOT / "kaggle_agent/dsh" / os.environ.get("AB_PATCH_CTRL", "aiplat.patch.yml")
PATCH_SLIM = ROOT / "kaggle_agent/dsh" / os.environ.get("AB_PATCH_SLIM", "aiplat.patch.yml")
STATE_SLIM = os.environ.get("AB_STATE_SLIM", "1")  # slim 组是否用 /state 瘦身
TASK = (ROOT / "kaggle_agent/dsh/TASK_FULL.md").read_text()
OUT = Path(__file__).parent / os.environ.get("AB_OUT", ".")
OUT.mkdir(exist_ok=True)
WALL = int(os.environ.get("AB_WALL", "1800"))
ARMS = [(g, arm, int(os.environ.get("AB_PORT0", "19100")) + i) for i, (g, arm) in enumerate(
    [(g, a) for g in os.environ.get("AB_GAMES", "r11l,ft09").split(",") for a in ("ctrl", "slim")])]

procs = []
for game, arm, port in ARMS:
    gs_env = dict(os.environ, A3_STATE_SLIM=STATE_SLIM if arm == "slim" else "0")
    gs = subprocess.Popen([sys.executable, "-m", "kaggle_agent.game_server", "--game", game,
                           "--port", str(port), "--max-actions", "200"],
                          cwd=ROOT, env=gs_env,
                          stdout=open(OUT / f"gs_{game}_{arm}.log", "w"), stderr=subprocess.STDOUT)
    procs.append(("gs", game, arm, port, gs))
time.sleep(10)
for game, arm, port in ARMS:
    ws = OUT / f"ws_{game}_{arm}"; home = OUT / f"home_{game}_{arm}"
    ws.mkdir(exist_ok=True); home.mkdir(exist_ok=True)
    env = dict(os.environ, DSH_HOME=str(home), DSH_PERMISSION_MODE="danger-full-access")
    patch = PATCH_SLIM if arm == "slim" else PATCH_CTRL
    d = subprocess.Popen(["node", str(DSH), "--profile", "headless", "--patch", str(patch),
                          TASK.replace("18999", str(port))],
                         cwd=ws, env=env,
                         stdout=open(OUT / f"dsh_{game}_{arm}.log", "w"), stderr=subprocess.STDOUT)
    procs.append(("dsh", game, arm, port, d))
    print(f"launched {game}/{arm} on :{port}", flush=True)

t0 = time.time()
while time.time() - t0 < WALL:
    if all(p.poll() is not None for k, *_, p in procs if k == "dsh"):
        break
    time.sleep(15)
results = []
for game, arm, port in ARMS:
    try:
        st = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/state", timeout=10).read())
        results.append(dict(game=game, arm=arm, level=st["level"], steps=st["steps_used"]))
    except Exception as e:
        results.append(dict(game=game, arm=arm, error=repr(e)))
for k, *_, p in procs:
    if p.poll() is None:
        p.terminate()
(OUT / "ab_results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
print(json.dumps(results, ensure_ascii=False), flush=True)
