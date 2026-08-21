# kaggle_agent — Kaggle 提交框架

第一次比赛提交的完整管线(2026-08-21 建, 本地已端到端验证)。

## 与 harness/ 的关系

**harness 的克隆体杠杆在真提交里不存在。** 隐藏游戏在网关(`ARC_BASE_URL`)后面,
每个动作都过 HTTP、计入 RHAE 分母 —— 没有 fork/peek/effect。所以本包只用
`EnvironmentWrapper.reset()/step()` 公开语义, 本地离线与真提交走同一条代码路径。
harness 想上场, 必须先把"从少量计分动作学世界模型"的路走通(Tycho/Duck 范式),
这是第二阶段接开源 LLM 的位置。

## 提交机制(从 Duck 第一名公开 notebook 逐 cell 核实)

- 提交物 = Kaggle notebook, 断网; 源码/权重以 Dataset 挂载;
  `arc-agi` 从比赛 wheelhouse 离线装。
- 真 rerun(`KAGGLE_IS_COMPETITION_RERUN=1`)走网关打隐藏游戏, 计分在网关侧
  scorecard(中途盲, 只能开一张); 平时 Save & Run 用公开环境文件离线跑。
- `ONLY_RESET_LEVELS=true` 必须在 import arc_agi 前钉死(RESET=关卡重置)。
- 每天限交 1 次; 提交动作永远人工拍板。

## 本地验证

```bash
# 主控直跑(两个游戏冒烟)
uv run python -m kaggle_agent.run_submission --env-dir environment_files \
    --games ft09,ls20 --seconds-per-game 45 --max-actions 300 --out-dir /tmp/smoke

# notebook 全流程(等价 Save & Run 离线路径)
A3_SECONDS_PER_GAME=6 A3_MAX_ACTIONS=120 uv run python - <<'EOF'
import json
ns = {}
for c in json.load(open("kaggle_agent/notebook/arc3-jinbo-submission.ipynb"))["cells"]:
    if c["cell_type"] == "code":
        exec("".join(c["source"]), ns)
EOF
```

## 发布(需要 kaggle CLI + ~/.kaggle/kaggle.json)

```bash
KAGGLE_USERNAME=<用户名> uv run python scripts/build_kaggle_bundle.py
kaggle datasets create -p dist/dataset    # 首次; 更新用 datasets version -m "..."
kaggle kernels push -p dist/kernel
# 网页: 打开 notebook -> Save & Run 验证 -> Submit to Competition
```

## 已踩过的坑(别再踩)

- **离线 scorecard 恒为 0 的假账**: `ONLY_RESET_LEVELS=true` 让首次 RESET 的
  `full_reset=False` -> `new_play` 不触发 -> 动作全漏记。竞赛模式服务端预建
  环境没这个问题; 离线靠 `_prime_offline_scorecard` 手工建卡, 且键必须用
  **带版本号的 game_id**(如 `ft09-0d8bbf25`), 裸 id 永远匹配不上。
- **一个坏游戏会炸整场**: 本地 dc22 环境文件残缺, `reset()` 返回 None 直接
  异常。主控已做单游戏隔离(记 ERROR 继续), 真提交必须保住这条。
- Kaggle 挂载点会因 owner/slug 冲突而变, 定位 bundle 用 marker 文件
  `arc3-jinbo-bundle.json` rglob, 不写死路径。

## 基线成绩(explorer v1, 零模型新颖度探索)

本地公开集: 3000 动作/游戏时 6 游戏过 2 关(ft09 L1 @1073 步, r11l L1 @113 步);
120 动作/游戏时 25 游戏过 2 关(lp85/r11l 各 L1)。定位是管线验证, 不是竞争力。
