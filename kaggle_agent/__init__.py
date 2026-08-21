"""Kaggle 提交子包: API-only 纪律下的比赛 agent。

与 harness/ 的根本区别(不要试图"复用"过来):

- **没有 fork/peek/effect**。真提交时隐藏游戏在网关(ARC_BASE_URL)后面,
  每个动作都过 HTTP、都计入 RHAE 分母, 克隆体不存在。harness 的一切
  "克隆体免费试错"路径在这里都是非法依赖。
- 只允许用 EnvironmentWrapper 的公开语义 reset()/step()。本地引擎内部
  (env._game / get_pixels / deepcopy)一概不碰 —— 本地离线跑和真提交
  必须是同一条代码路径, 否则离线验证不了任何东西。
- 同一份代码双模式: KAGGLE_IS_COMPETITION_RERUN/ARC_BASE_URL 存在时走
  COMPETITION(网关), 否则走 OFFLINE(本地公开环境文件)。

模块分工:
- remote_env.py   ApiGame: wrapper 之上的最小句柄, 动作计数记账
- explorer.py     第一发 agent: 预算内新颖度探索(零模型, 纯符号)
- run_submission.py 主控: 建 Arcade/scorecard, 逐游戏跑, 汇总落盘
"""
