"""(08-25 副本: 对照组 /state 也可瘦身 AB_STATE_CTRL; 本地 game_server 请求绕过代理) /state 瘦身 A/B: 对照组=现状(frame+summary), 实验组=A3_STATE_SLIM=1(仅summary, 原文靠/grid)。
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
TASK_CTRL = (ROOT / "kaggle_agent/dsh" / os.environ.get("AB_TASK_CTRL", "TASK_FULL.md")).read_text()
TASK_SLIM = (ROOT / "kaggle_agent/dsh" / os.environ.get("AB_TASK_SLIM", "TASK_FULL.md")).read_text()
OUT = Path(__file__).parent / os.environ.get("AB_OUT", ".")
OUT.mkdir(exist_ok=True)
WALL = int(os.environ.get("AB_WALL", "1800"))
SEEDS = int(os.environ.get("AB_SEEDS", "1"))
ARMS = [(g, arm, sd, int(os.environ.get("AB_PORT0", "19100")) + i) for i, (g, arm, sd) in enumerate(
    [(g, a, sd) for g in os.environ.get("AB_GAMES", "r11l,ft09").split(",")
     for a in os.environ.get("AB_ARMS", "ctrl,slim").split(",") for sd in range(SEEDS)])]

procs = []
for game, arm, sd, port in ARMS:
    gs_env = dict(os.environ, A3_STATE_SLIM=STATE_SLIM if arm == "slim" else os.environ.get("AB_STATE_CTRL", "0"), no_proxy="127.0.0.1,localhost", NO_PROXY="127.0.0.1,localhost")
    gs = subprocess.Popen([sys.executable, "-m", "kaggle_agent.game_server", "--game", game,
                           "--port", str(port), "--max-actions", "400"],
                          cwd=ROOT, env=gs_env,
                          stdout=open(OUT / f"gs_{game}_{arm}{sd}.log", "w"), stderr=subprocess.STDOUT)
    procs.append(("gs", game, arm, port, gs))
time.sleep(10)
for game, arm, sd, port in ARMS:
    ws = OUT / f"ws_{game}_{arm}{sd}"; home = OUT / f"home_{game}_{arm}{sd}"
    ws.mkdir(exist_ok=True); home.mkdir(exist_ok=True)
    env = dict(os.environ, DSH_HOME=str(home), DSH_PERMISSION_MODE="danger-full-access", no_proxy="127.0.0.1,localhost", NO_PROXY="127.0.0.1,localhost")
    patch = PATCH_SLIM if arm == "slim" else PATCH_CTRL
    d = subprocess.Popen(["node", str(DSH), "--profile", "headless", "--patch", str(patch),
                          (TASK_SLIM if arm == "slim" else TASK_CTRL).replace("18999", str(port))],
                         cwd=ws, env=env,
                         stdout=open(OUT / f"dsh_{game}_{arm}{sd}.log", "w"), stderr=subprocess.STDOUT)
    procs.append(("dsh", game, arm, port, d))
    print(f"launched {game}/{arm}#{sd} on :{port}", flush=True)

t0 = time.time()
while time.time() - t0 < WALL:
    if all(p.poll() is not None for k, *_, p in procs if k == "dsh"):
        break
    time.sleep(15)
results = []
for game, arm, sd, port in ARMS:
    try:
        st = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/state", timeout=10).read())
        results.append(dict(game=game, arm=arm, seed=sd, level=st["level"], steps=st["steps_used"]))
    except Exception as e:
        results.append(dict(game=game, arm=arm, seed=sd, error=repr(e)))
for k, *_, p in procs:
    if p.poll() is None:
        p.terminate()
(OUT / "ab_results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
print(json.dumps(results, ensure_ascii=False), flush=True)
