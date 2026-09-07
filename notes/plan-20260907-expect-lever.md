# 下一刀：动作带预测（Retrodict 的 expect），实现路径已探明

## 为什么是它
- Retrodict（社区榜第 2, 99.86% RHAE, $654）把「每个动作必须带预测，没有预测的动作定义为浪费动作」
  写成铁律，用 7703 步打完 183 关；我们 4902 步只过 48 关。
- baseline1 paper02（arXiv 2607.15439）的消融：verification 这一档的净增量在**低 effort 上最大**
  （gpt-5.5 high +7.29 / gpt-5.4 high +8.56；xhigh 上只有 +1.69 / +0.62）。
  我们的 Flash-Next 属于推理预算紧的那一类，且 08-27 自测「思考封顶是负收益」。
- 🚨 裸的可执行世界模型（ewma）在五组配置里输给纯文本世界模型三组 —— 收益不在「可执行」四个字，
  在后面挂的**简化 + 验证**。所以要做就做验证这一档。

## 实现点（已在 v13 底子上确认）
1. `_normalize_python_actions`（tool_agent.py:1691）：让 action dict 多收一个 `expect` 字段，
   形如 `[[row, col, color], ...]`，校验范围 0..63 / 0..15。
2. `solver.py:1958` 的批次循环**已经有完整腰斩机制**（`stop_reason` + `break`，现有 7 个停止条件：
   batch_checkpoint / stopped / invalid_action / action_error / run_complete / game_over / level_completed）。
   加一条即可：
   ```python
   if expect and 实际棋盘 != 预测:
       stop_reason = "prediction_mismatch"
       break
   ```
3. 失配详情（哪个格子、预测什么、实际什么）回灌到 `stop_detail`，
   它已经会进 user prompt（`_build_user_prompt` 里的 "Why the previous sequence stopped"）。

## 风险
- harness 支持了但模型不填 `expect` 就等于没做 → 必须配提示词，而提示词改动风险高
  （08-31 两个文案改动双双负收益，「给模型的每句话都会被当真并放大」）。
- 所以提示词那句要极度克制，只陈述机制（"每个 action 可带 expect；不符会停下并告知差异"），
  不要写价值判断（不要写"没有预测的动作是浪费"这种，容易被放大成瘫痪）。
- 按三层框架验：DeepSeek 先看模型填不填 expect、填得准不准；再上 qwen3.6-plus；最后 Kaggle。

---

# winframe 的已知设计权衡（09-07 定，明天可调）

触发条件是 `previous_step_summary["level_transition"]`，**只在刚过关那一轮为真**，
所以赢帧只送达一次（成本 ≈1051 token/次，一局过 2 关就是 2 次，可控）。

三个待验的调节旋钮：
1. **停留轮数**：只给 1 轮模型可能来不及消化。`_solved_frame_from_history` 看的是 history 末 3 条，
   窗口本身够，卡的是 level_transition 只真一轮。给 2~3 轮会让成本翻倍，需要 A/B。
2. **压缩表示**：行内 RLE + 相同行合并能压到 189 token（原文 1039，18%），
   但换格式就是新变量——模型习惯读 `current_frame` 那种明文网格，压缩格式未必认。
3. **只给 diff**：赢帧 vs 赢之前那一帧的差异格子，比全盘更聚焦「什么变化导致了赢」，
   也更省。但会丢掉「成功状态整体长什么样」这个信息。

优先级：先看今晚隐藏集出分，再决定调哪个。三个都要单独 A/B，别一起上。

---

# 模型侧草稿（明天验，别直接上）

## 工具 schema 里加的字段描述（事实陈述，无价值判断）
```
expect: optional. A list of [row, col, color] cells the board is expected to show
        after this action settles. When any listed cell differs, the remaining
        actions in the queue are not executed and you are told which cells differed.
```

## 系统提示词里加的一句（候选，二选一，各自 A/B）
- A（纯机制，最克制）：
  `Each action in action([...]) may carry "expect": [[row, col, color], ...]. Cells that
   disagree stop the rest of the queue and report the difference.`
- B（A + 一句用法）：
  A 的内容 + `Use it on actions whose effect you believe you can predict.`

🚨 **不要写**的措辞（08-31 实证会被放大成灾难行为）：
- 「没有预测的动作是浪费」→ 会让模型不敢出手（试按钮说明书那次 ft09 三遍只走 36/16/13 步全程空想）
- 「零成本」「尽量多预测」→ 会让模型乱试烧步（计分说明那次 cd82 烧到 500+ 步）
- 任何带价值判断或禁令的句子。只陈述机制，让模型自己决定用不用。

## 验收指标（第一层 DeepSeek）
1. **模型填不填**：带 expect 的动作占比（0 就说明 schema 描述没被看见，要改描述而不是加压）
2. **填得准不准**：prediction_mismatch 触发率（太高说明模型乱猜，太低说明它只在稳赢时才填）
3. **有没有换来过关**：过关数对比——这才是终判据，前两个只是过程指标

---

## 09-07 傍晚 第一轮实验：只在工具描述里说，模型不用

改动：`_PYTHON_TOOL_DESCRIPTION` 加 126 字符，纯机制陈述
（"An action object may also carry `'expect': [[row,col,color],...]` -- the cells the settled
board must show after that action. A cell that disagrees stops the remaining queued actions
and reports which cells differed."），系统提示词一字未动。

**结果（ft09 单局 10 分钟，DeepSeek，ARC3_WINFRAME=0）：22 段去重代码、6 次 `action()` 调用、
`expect` 出现 0 次。模型一次都没用。**

与 [[arc3-model-never-reads-history]] 同一个模式：**告诉模型"有这个东西"不足以让它用**。
第二轮改为补一句用法陈述（何时该用），仍不写价值判断。

---

## 第三轮的两个方向（等第二轮结果定）

### 方向 A：把说明挪进系统提示词
风险最高的位置。只在工具描述已证明不够时才用，且措辞照旧只陈述机制+用法，不写价值判断。

### 方向 B（更可能对我们的模型成立）：转向"回放验证历史"而不是"预测未来"
**假设**：expect 要求模型预测**下一帧**，这对 DeepSeek/Flash-Next 这类弱模型太难——
Retrodict 用的是 gpt-5.6-sol max effort。而 baseline1 的 verification 是反向的：
不预测未来，只要求**解释已经发生的历史**，有标准答案可对，弱模型能迭代改进。

**形态**（= 计划里的第 4 刀，`notes/plan-20260903-five-levers.md` 已写死）：
- 沙箱新增持久对象 `world_model_code`（跨轮保留的代码串）
- `replay(code)`：用已记录的 `transitions` 逐条回放，返回命中率 + 首个失配位置
- user prompt 里加一行「当前世界模型回放命中率 x/y，失配在第 k 步」
- 命中率 100% 才允许用它规划

**为什么它比 expect 更适合我们**：
| | expect(Retrodict) | replay(baseline1) |
|---|---|---|
| 要求模型做什么 | 预测**下一帧**的格子 | 写函数**解释历史**转移 |
| 错了的代价 | 浪费一个动作 | 零（纯离线回放） |
| 有无标准答案 | 无（要等执行才知道） | 有（历史帧就是答案） |
| 对弱模型 | 难 | 可迭代 |

🚨 计划里已记的坑：裸预测命中率只有 48.6% 且证据越多越差（08-30 测），
所以回放验证必须是**硬门**——命中率不到 100% 的世界模型只当假设，不当规划器。
