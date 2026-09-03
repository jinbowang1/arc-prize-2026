# 五刀计划（2026-09-03 定，起点 = 第十一发 5.49 / 榜第 2）

规矩：一天只改一样；给模型看的改动先本地 DeepSeek 验、再公开集跑；提交由用户拍板；同配置不重投。
GPU 额度：按账号 30h/周，周五 08:00（北京）刷新。一次公开跑约 2.6h（setup 0.3h + 2.2h 打局 + 收尾）。
底子：`reference/fnrep-source-v4/`（= 上传数据集 `jinbowang1/arc3-fnrep-source`）。
每一刀都从这个底子拷一份新目录 `build_v12_*`，改完传成**新名字**的数据集 + **新名字**的 kernel（同名连死两次挂载就换名的老规矩）。

健康检查三件套（每次公开跑必查，缺一不可）：25 局全结束 / 零崩 / 输出目录里有 `submission.parquet`。

---

## 第 1 刀：把跨局管家喂活（半天本地 + 1 次公开跑）

**现状**：管家读 `/kaggle/working/*_requests.jsonl`，该文件只在 `save_request_logs=True` 时写；Son 的 pickle 里是 False。整场 0 证据、账本空、每轮塞 352 字符空块。

**改法（选 B）**
- A. 拨开关：notebook 第 12 单元 `bm.solver.save_request_logs = True`。每次请求把整段消息落盘，25 局 2.2h 估几百 MB 到 GB 级，收尾必须删。不选。
- B. 改管家证据源：`nvfp4_cross_game_curator.py::collect_world_models()` 改读 `<events-dir>/prompts/*.log`。
  这个文件每轮由 `tool_agent._write_prompt_log_snapshot` 覆盖写，内容是渲染后的「最新一次模型输入」，
  随身世界模型那段（`Working world model carried from earlier turns: … End of carried world model.`）原样在里面，
  现有正则 `WORLD_MODEL_RE` 直接命中；`Current state:` 行也在。抠不到时退而取文件里最后 3 个 `[ASSISTANT]` 块。
  文件名 `<game>_p0.log` → game_id 取前缀。改动约 40 行，管家其余逻辑不动。

**改哪些文件**：只有 `nvfp4_cross_game_curator.py`。伺服脚本、agent 不动。

**本地验证（不用 GPU）**
1. 从公开跑 kernel 输出把 `prompts/` 目录拉回（`kg.sh me kernels output … --file-pattern "prompts/*"`），当作离线证据。
2. 管家 `--base-url` 指向 hub 的 DeepSeek，`--poll-seconds 2`，跑 3 分钟：
   验收 = `health.json` 的 `games_seen ≥ 3`、`requests_completed ≥ 1`，`ledger.json` 的 `themes` 非空且每条 ≤ 45 词。
3. 把 ledger 塞进 `tool_agent._common_themes_prompt_block()` 干跑一次，看注入块 ≤ 6000 字符、≤ 12 条。

**公开跑验收**：`world-model-curator/health.json` requests_completed > 0；`gameplay-injections.jsonl` 里 injected_theme_count > 0 的行占比；均分不低于 9 上下（单次方差大，只看「不明显掉」）。

**风险**：管家占用伺服的一路生成（28 局 + 管家 = 29 路争 22 路）；ledger 里出现错误规律会带偏 28 局。
缓解：`--max-entries 6` 已封顶；注入块里自带 caution 字段。

**提交条件**：账本填上 + 均分不掉 → 可作第十二发候选。

---

## 第 2 刀：把我们的第一关补丁叠到 Son 的 agent 上（1 天本地 + 1 次公开跑）

**素材**（在 `build_l1` 包里，对 Tufa 干净版的 diff：solver +442 / tool_agent +57 / prompts +20）
- 告诉模型怎么计分：`LEVEL_PRIOR_LINE`（控制方式 93% 跨关沿用，布局从不沿用）+ 计分公式说明。
- 开局替模型试按钮：`DUCK_PROBE_MANUAL`，solver 在每关开局把每个合法动作各按一次、分类效果（平移/变色/无变化/瞬态），写成说明书经 sidecar 给 tool_agent，每关只给一次，上限 14 步。
- 通关画面转写：`DUCK_WINFRAME`，过关那一步引擎返回两层画面，把「赢下来的棋盘 vs 下一关开局」的差异写成一句事实。27B 上效果在噪声内，**这次先不叠**，留作第 2b。

**移植难点**：Son 和我们改的是同一批函数（`_build_user_prompt`、`_execute_action`、`play`）。
不能三方合并，要手工把我们的三段逻辑搬进 Son 的版本：
- solver：探测函数 5 个 + `_write_duck_extra` sidecar + 开局探测调用点（放在 `play()` 首轮之前）
- tool_agent：`_load_duck_extra` + `_build_user_prompt` 里两处注入（放在管家块之后、随身世界模型之前）
- prompts：加 `LEVEL_PRIOR_LINE`

**与 Son 机制的冲突点要先想清**：
- 开局探测的 14 步会进动画感知的「前 5 步热身」窗口，探测动作的帧数会当基线。可接受，探测本身就是普通动作。
- 探测那 14 步计入该关步数，按平方计分会亏分。27B 上这笔账算过（探测省下的瞎试步数 > 14），Flash-Next 更聪明，瞎试更少，**这笔账要重算**：公开跑对比「开张数、第一关平均步数」。
- 8 步封顶与探测无关（探测走 `_execute_action` 直调，不经 `step_env`）。

**本地验证**：DeepSeek 跑 3 局（ft09 / ls20 / r11l）各 2 遍，看说明书内容正确、注入只一次、动画感知不因探测误报。
**公开跑验收**：开张数 23 → 目标 25；第一关平均步数不升；均分不掉。
**提交条件**：开张数涨且均分不掉。

---

## 第 3 刀：伺服提速（半天 + 1～2 次公开跑）

**3a 多词预测投机解码**
- 模型自带 1 层 MTP（BF16 原样保留，`mtp_enabled: false` 是 Son 主动关的，原因没写）。
- 改 `kaggle_flashnext_setup.py::launch_server()`（走包装器字符串替换，底层文件不动）：加
  `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`；起不来就退回原命令（照抄 V31 的 primary/fallback 写法）。
- 不确定项：这个 nightly vLLM 对 qwen4_exp 架构的 MTP 支持；PLE 卸载 + MTP 的显存账。**只能上 Kaggle 试**，本地没有 Blackwell。
- 验收：`model-smoke.json` 的 22 路 wall_seconds 变短；公开跑总步数 4902 → 目标 6000+；均分不掉（27B 上 MTP3 隐藏集约 +0.17）。

**3b 并发对齐** —— ⚠️ 先算批次账再动
- 现状 28 路客户端 / 22 路伺服。总跑时 = 0.27h + ceil(N/并发) × 2.2h，N = 隐藏局数（未知；第十一发 submit 后 >9h 才出分，PENDING 含排队，推不出 N）。
- 并发 28 能出分是唯一事实。并发降到 22 会让批次数变多（N=67~84 时 3 批→4 批，多 2.2h）；9h 管不管重跑不知道，别拿名额去试。
- 所以并发**只能往上调**（比如 max-num-seqs 22 → 28 对齐客户端，前提是 KV 显存够；`--gpu-memory-utilization 0.96` 已顶满，可能要靠 `--kv-cache-dtype fp8` 换空间）。
- 验收：启动不 OOM；`model-smoke.json` 28 路热身通过；公开跑均分不掉。

**提交条件**：总步数涨 20% 以上且均分不掉。

---

## 第 4 刀：可执行世界模型（2～3 天，按三层框架）

**方法**（arXiv 2605.05138，代码 github.com/astroseger/arc-3-agents-baseline1）：agent 维护一段可运行的 Python 世界模型；
每轮先用历史转换回放验证它（预测的下一帧 == 真实下一帧），验证过的模型才拿来做规划（在模型里搜动作序列）；
不符就重构（奥卡姆偏置，改最少）。GPT-5.5 公开局通关 15/25，GPT-4.5 8/25，天花板随模型涨。

**落到 Son 的 harness 上的形态**（结构承担约束，模型只出理解）：
- 沙箱新增两个持久对象：`world_model_code`（跨轮保留的代码串）和 `replay(code)`（用 `transitions` 逐条回放，返回命中率与首个失配）。
  注意沙箱每次从零起，持久性要靠 tool_agent 侧保存并在 `_serialized_runtime_state` 里回灌。
- 用户消息里加一段「当前世界模型回放命中率 x/y，失配在第 k 步」。
- 系统提示词加一条：先修世界模型再出手；命中率 100% 时用 `plan()` 在模型里搜。
- 8 步封顶、动画感知照旧。

**三层验证**
1. DeepSeek 本地：4 个已满分游戏（ls20/tr87/ft09/cd82）看回放命中率能否到 100%、规划出的序列能否过第一关。
2. 27 上 qwen3.6-27b：25 局公开集，对比无世界模型版。
3. Kaggle Flash-Next：公开跑一次。
**验收**：公开集关数 48 → 目标 60+。**提交条件**：关数涨且开张不掉。

**已知的坑**：裸预测命中率只有 48.6% 且证据越多越差（08-30 测），所以回放验证必须是硬门，验证不过的模型只当假设不当规划器。

---

## 第 5 刀：修两个隐患（1 小时，随任意一刀一起上）

**5a 看门狗**：notebook `_v31_spawn` 读 `qwen38-model-provenance.json`（不存在），真挂时复活必失败。
改法：去掉重启逻辑，只保留「连挂 3 次 → 置 stop_event 保住已得分」。因为 Flash-Next 起一次要 16 分钟且要重挂 PLE，重启本身不划算。

**5b 软截止**：结论是**现在不装**。
- 这一发 submit 后超过 9 小时才出分（守望日志 5.4h 仍 PENDING，次日早上见分）。PENDING 含排队，Kaggle 不给完成时间，所以运行本身多长、9h 是否管重跑，都没证据。能确定的只有：当前配置能出分。
- 真提交路径上 `submission.parquet` 由 Kaggle 的 gateway 按 scorecard 生成，我们代码不写它；软截止触发的取消链（`_cancel_at` → 取消 solver 任务 → 120s drain → `finish_remaining`）在比赛模式下**从没跑过**，装上去反而引入一条未测路径。
- 什么时候必须装：第 3b 刀动了并发、或第 4 刀让每轮变慢导致某批跑不满……凡是让「批次数」或「setup 时长」变大的改动，先算 0.27h + ceil(N/并发) × 2.2h，N 取 84 保守值，超 8.5h 就必须装并在公开跑里模拟触发一次（把 soft_end 设成开跑 + 40 分钟看 parquet 是否还在）。

---

## 排期

| 日期 | 做什么 | GPU |
|---|---|---|
| 09-03 周三 | 第 1 刀本地改 + DeepSeek 验；第 5a 改 notebook | 0 |
| 09-04 周四 | 第 2 刀移植 + DeepSeek 3 局验；第 3a 包装器改好干跑 | 0 |
| 09-05 周五 08:00 额度刷新 | 公开跑 A = 第 1 刀 + 5a（2.6h）；通过 → 当天提交候选 | 2.6h |
| 09-05 下午 | 公开跑 B = A + 第 2 刀 | 2.6h |
| 09-06 周六 | 公开跑 C = B + 第 3a；若 3a 起不来自动退回 | 2.6h |
| 09-07 起 | 第 4 刀三层验证（DeepSeek → 27 → Kaggle） | 2.6h/次 |

每天 1 发名额给「最新通过健康检查且均分不掉」的版本；哪一发投由用户拍板。

---

## 09-03 进展（第 1 刀本地完成，等周五 GPU）

**做了什么**：`build_v12_curator/`（从 v4 底子拷出，内置 git，基线=逐字节底子）
- `nvfp4_cross_game_curator.py`：证据源改读 `<events-dir>/prompts/*.log`（每轮覆盖写的最新模型输入，永远存在）；
  解析器 `parse_prompt_log()` 拆 `[MODEL INPUT]` 段的 `[USER]/[ASSISTANT]/[REASONING]/[TOOL RESULT]` 块；
  新增 `--min-support-games 2` 硬门；`--api-key/--plain-openai/--prompt-logs-dir` 只为本地验证。
- `dataset-metadata.json` → `jinbowang1/arc3-fnrep-v12-curator`（新名）。
- kernel 副本 `kaggle_agent/duck/kernels/k_v12_curator/`（slug `arc3-fnrep-v12-curator`），挂载名同步改；
  第 5a 刀一并做了：`_v31_recover` 短路成「不重启、不杀伺服、直接 return False → 置 stop_event 保分」。

**🚨 新发现（比 09-03 解剖更深一层）**：FN 公开跑 22 局 prompts 日志里，「Working world model carried from earlier turns:」
一次都没出现（0/22），`World model:/Goal model:` 前缀 0 局——**Son 的随身世界模型机制在 Flash-Next 上本来就是空的**，
模型的信念只在每次工具调用前的一句助手短句（~100 字符）和推理流（几千字符）里。
所以就算拨 `save_request_logs` 开关，管家拿到的也是同样的东西。现在的证据 = 状态行 + 最近 3 条助手短句 + 最新推理尾 3000 字符。

**本地验证结果**（DeepSeek 官方 `deepseek-v4-flash`，关思考，22 局离线证据）：
- 收集：22/22 局抓到；一次请求打包 10 局 = 10.5K prompt token，7.9 s 回 6 条。
- 质量：**6/6 条 support_games=1**，全是单局事实（"dam/water 机制"、"3x3 glyph 编码"…），不是跨局规律。
  → 加硬门后账本 0 条，注入块退回「暂无」353 字符，与线上现状一致。
- 注入干跑（Son 的 `_common_themes_prompt_block`）：6 条时 3286 字符 ≤ 5000 上限；0 条时 353 字符。

**收益预期下调**：管家能不能在 2.2h 内产出 ≥2 局证据的跨局规律，只有 Flash-Next 当管家跑一次公开集才知道；
硬门保证最坏情况 = 现状（账本空），不会更差。公开跑验收改为：`health.json` requests_completed > 0；
`ledger-revisions.jsonl` 里 entry_count 的分布；有条目时 `gameplay-injections.jsonl` injected_theme_count > 0 的占比；均分不明显掉。

**顺手修正**：kaggle CLI 比赛 slug 是 `arc-prize-2026-arc-agi-3`（`arc-agi-3` 会 404/403，不是鉴权坏）。
