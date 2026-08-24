"""把一局游戏挂成本地 HTTP 服务, 供外部 agent(如 deepseek-harness)驱动。

观察层复用 repl_agent 的全套资产: 全图渲染/连通块摘要/颜色账本/计数器去噪/
对象级 diff —— 外部 agent 吃现成摘要, 不用自己解析 4096 个数字。

两种用法:
- 独立进程(本地攻关):
    uv run python -m kaggle_agent.game_server --game r11l --port 18999 --max-actions 150
- 进程内多实例(赛场并发): `GameServer(game, max_actions).start(port)` 一局一个端口,
  同一进程里 N 局同时挂着, 共用一个 Arcade/scorecard(竞赛模式只准开一张卡)。

接口(全 GET, 返回 JSON):
    /state              当前局面(渲染+摘要+账本)
    /act?a=6&x=40&y=24  执行动作(a=1..5按键, a=6点击带x,y); 返回效果摘要
    /grid               裸 grid(list[list[int]]), 供脚本分析
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .llm_agent import _grid_summary
from .remote_env import Action, ApiGame
from .repl_agent import _counter_cells, _diff, _full_frame, _obj_diff


def _account(grid, base):
    cc = Counter(v for row in grid for v in row)
    parts = []
    for c in sorted(set(cc) | set(base)):
        n0, n1 = base.get(c, 0), cc.get(c, 0)
        if max(n0, n1) < 1500:
            parts.append(f"色{c} {n0}→{n1}" if n0 != n1 else f"色{c} {n1}")
    return ", ".join(parts)


class GameServer:
    """一局游戏的观察层 + HTTP 壳。状态全在实例上, 多局并发互不串。"""

    def __init__(self, game: ApiGame, max_actions: int):
        self.game = game
        self.max_actions = max_actions
        self.obs = game.reset()
        self.history: list = []
        self.base_cc = dict(Counter(v for row in self.obs.grid for v in row))
        self.t0 = time.time()
        self.per_level_steps: list[int] = []
        self._level_start = game.steps
        self._lock = threading.Lock()  # 同一局的 act 串行, 观察层账本不许交错
        self._httpd: ThreadingHTTPServer | None = None

    # ---- 观察 ----
    def payload_state(self) -> dict:
        o = self.obs
        ctr = _counter_cells(self.history)
        d = {
            "level": o.level, "win_levels": o.win_levels,
            "steps_used": self.game.steps, "steps_budget": self.max_actions,
            "done": o.done, "state": o.state,
            "frame": _full_frame(o.grid),
            "summary": _grid_summary(o.grid),
            "color_account": _account(o.grid, self.base_cc),
        }
        if ctr:
            d["counter_cells_hint"] = (f"计数器格{len(ctr)}个(如{sorted(ctr)[:8]}): 每步自动变化,"
                                       " 与动作内容无关, 分析时剔除")
        return d

    # ---- 行动 ----
    def do_act(self, a, x, y) -> dict:
        game = self.game
        if game.steps >= self.max_actions:
            return {"error": "动作预算已用尽", "steps_used": game.steps}
        prev = self.obs
        action = Action.click(int(x), int(y)) if int(a) == 6 else Action.key(int(a))
        cur = game.act(action)
        resp = {"level": cur.level, "steps_used": game.steps, "done": cur.done}
        if cur.dead:
            self.history.append((repr(action), prev.grid, cur.grid))
            cur = game.reset_level()
            resp.update(dead=True, note="导致死亡, 已重置回本关起点", level=cur.level)
            self.obs = cur
            return resp
        self.history.append((repr(action), prev.grid, cur.grid))
        self.obs = cur
        if cur.level != prev.level or cur.done:
            self.per_level_steps.append(game.steps - self._level_start)
            self._level_start = game.steps
            self.base_cc = dict(Counter(v for row in cur.grid for v in row))
            resp.update(level_up=True,
                        note=f"🎉 过关! 现在 level {cur.level}/{cur.win_levels}"
                             + (" 全部通关!" if cur.done else ""))
            return resp
        ctr = _counter_cells(self.history)
        resp["effect"] = _obj_diff(prev.grid, cur.grid, ctr)
        resp["changed_cells"] = len(_diff(prev.grid, cur.grid))
        return resp

    def handle(self, path: str, q: dict) -> dict:
        with self._lock:
            if path == "/state":
                return self.payload_state()
            if path == "/grid":
                return {"grid": [[int(v) for v in row] for row in self.obs.grid]}
            if path == "/act":
                return self.do_act(q.get("a", 6), q.get("x", 0), q.get("y", 0))
            return {"error": f"unknown path {path}", "paths": ["/state", "/act", "/grid"]}

    # ---- HTTP 壳 ----
    def start(self, port: int) -> "GameServer":
        """后台线程起服务, 立即返回。"""
        server = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # 静音访问日志
                pass

            def do_GET(self):  # noqa: N802
                u = urlparse(self.path)
                q = {k: v[0] for k, v in parse_qs(u.query).items()}
                try:
                    body = server.handle(u.path, q)
                except Exception as e:  # noqa: BLE001
                    body = {"error": repr(e)}
                data = json.dumps(body, ensure_ascii=False).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self._httpd = ThreadingHTTPServer(("127.0.0.1", port), H)
        self._httpd.daemon_threads = True
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None


def main():
    from .run_submission import _build_arcade

    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--port", type=int, default=18999)
    ap.add_argument("--max-actions", type=int, default=150)
    ap.add_argument("--env-dir", default="environment_files")
    a = ap.parse_args()
    arcade = _build_arcade(a.env_dir)
    gid = next(i.game_id for i in arcade.get_environments() if i.game_id.startswith(a.game))
    gs = GameServer(ApiGame(arcade.make(gid), gid), a.max_actions).start(a.port)
    print(f"game_server: {gid} on :{a.port}, budget {a.max_actions}", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        gs.stop()


if __name__ == "__main__":
    main()
