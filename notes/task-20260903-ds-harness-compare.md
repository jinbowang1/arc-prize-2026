# 任务书：用 DeepSeek 同题对照「Son 的 harness」与「我们之前的 harness」

写于 2026-09-03，交给 Opus 执行。本文里的每条路径、行数、配置项都在 09-03 当场核过一遍，可以直接照着做。
遇到与本文不符的现场事实，以现场为准，并在报告里写明哪一条对不上。

---

## 一、一句话任务

在本机零 GPU 的条件下，让同一个模型（hub 上的 DeepSeek V4 Flash）分别跑两套 harness，
在同一批公开局上打同样的时长，看两套 harness 本身带来的差别有多大、差在哪里。

## 二、为什么做这件事

用户 09-03 原话：「我觉得现在你用 deepseek 去跑一遍 son 的 harness，然后我们来看看和之前的 harness 的区别吧。」

背景是这样的：

- 我们第十一发把 Son Pham 的整包逐字节复刻上线，隐藏集从 2.01 跳到 5.49，一度排第 2。
  09-03 下午已经掉到第 3（seele 5.53 在前，Fususu 5.43 在后），三队差距在噪声范围内。
- 这一跳同时换了两样东西：模型从 Qwen3.8-27B 换成 Flash-Next 125B，harness 从我们改过的 Duck 换成 Son 的版本。
  两样一起换，功劳分不清。公开 25 局均分从 4.69 涨到 9.11 也是两样一起变的结果。
- 用户对千问一直有怀疑（原话「我总感觉千问好像不太好使」）。09-03 的调研结论是 96GB 单卡内没有
  比 Flash-Next 更强的开源模型可换，所以下一步的空间在 agent 侧。
- 要判断 agent 侧还有多少空间，先得知道：Son 的 harness 相对我们之前那套，到底强在哪、强多少。
  固定模型、只换 harness，是唯一能把这个数拆出来的做法，而且零 GPU 就能做。

判据（报告必须正面回答这三个）：

1. 同模型同题下，两套 harness 的过关数、步数效率、单局用时差多少。给出逐局表，不给平均值糊过去。
2. 差别是由哪几处代码带来的。要落到具体文件和具体机制，不能停在「Son 的更好」。
3. 这些机制里，哪些是我们能在不换模型的前提下搬过来的，按性价比排序。

## 三、两个 harness 的身份

两套其实是同一个代码库的两个版本，都源自 Tufa Labs 的 Duck harness。

| | 我们之前用的 | Son 的 |
|---|---|---|
| 路径 | `reference/duck-harness/ARC3-Inference` | `reference/fnrep-source-v4/src/ARC3-Inference` |
| 来源 | Tufa 官方公开分享包，加上我们 08-25 的两处本地改动 | Son Pham 的 Kaggle 包 v4，第十一发上线的就是它 |
| 配套框架 | `reference/duck-harness/tufa-arc-agi-framework` | `reference/fnrep-source-v4/src/tufa-arc-agi-framework` |
| 本地环境 | `.venv` 已装好，Python 3.12.12，`import taaf, arcengine, arc_agi` 全通 | 尚未建 venv |

已核实的文件级差异（`diff` 出的增删行数）：

| 文件 | 旧版行数 | Son 版行数 | 差异行数 |
|---|---|---|---|
| `inference/framework/solver.py` | 1715 | 2680 | 1843 |
| `inference/agent/tool_agent.py` | 2435 | 2527 | 918 |
| `inference/agent/python_tool_sandbox.py` | 628 | 873 | 349 |
| `inference/agent/prompts.py` | 137 | 75 | 100 |
| `inference/framework/run.py` | 1374 | 1422 | 92 |
| `inference/utils/openai_compat.py` | 77 | 72 | 5（全是我们自己的改动） |

Son 版另有两个新文件 `inference/utils/rearc_baselines.py` 和 `rearc_version.py`。

大头在 `solver.py`（多出约 1000 行）。09-03 的解剖报告 `notes/fnrep-v4-anatomy-20260903.html` 第五节
已经指认其中约 1300 行是动画感知相关，可以先读那一节，别从零重读。

`openai_compat.py` 的 5 行差异是我们 08-25 打的补丁，内容见 `kaggle_agent/duck/hub-thinking-disabled.patch`：
`HUB_THINKING_DISABLED=1` 时给请求体加 `thinking: {"type": "disabled"}`。这个补丁要原样移植到 Son 版，
否则 hub 上的 DeepSeek 默认开思考，会把 token 预算全烧在推理上（09-03 本地跑管家时踩过，报 max_tokens 用尽）。

## 四、现有可复用资产

全部已核实存在：

- `environment_files/`（3.7MB，25 个公开局：ar25 bp35 cd82 cn04 dc22 ft09 g50t ka59 lf52 lp85 ls20 m0r0 r11l re86 s5i5 sb26 sc25 sk48 sp80 su15 tn36 tr87 tu93 vc33 wa30）。
  离线打局靠它，不需要联网访问 three.arcprize.org。
- `kaggle_agent/duck/inference.hub-ds.json`：旧 harness 接 hub DeepSeek 的完整配置，可直接当模板。
  关键几项是 `provider: openrouter`、`base_url: http://llm-model-hub-apis.sf-express.com/v1`、
  `environment.environments_dir` 指向本地 `environment_files`、`multimodal.context: ""`（关图）、
  `analyzer.max_output: 16000`、`analyzer.timeout: 300`。
- `kaggle_agent/duck/run_duck_vc33.sh`：可运行的跑法样板。注意它做了三件事，缺一不可：
  从 `~/Desktop/2-快件质量与OpenClaw/langfuse-local/.env` 读 `AIPLAT_KEY` 当 `OPENROUTER_API_KEY`、
  设 `HUB_THINKING_DISABLED=1`、`unset` 掉全部 proxy 变量。
- `~/.config/arc3/deepseek_official.key`：DeepSeek 官方 key，走官方 API 时用，需要 Clash 代理 `127.0.0.1:6789`。
  仅作 hub 不可用时的备选，不要两个通道混用。
- `probes/20260903/fn_v4_out/prompts/`：Flash-Next 公开跑 22 局的 prompt 快照，用来对照 Son harness 实际喂给模型的东西长什么样。
- 报告样式参考 `notes/fnrep-v4-anatomy-20260903.html`。

## 五、环境搭建

Son 版的 venv 还没建。`uv 0.11.0` 在 `/opt/homebrew/bin/uv`，Python 3.12.12 已有。

```
cd ~/Desktop/project/arc-agi-3/reference/fnrep-source-v4/src/ARC3-Inference
uv venv --python 3.12.12 .venv
env -u VIRTUAL_ENV uv sync --locked --all-extras --python .venv/bin/python
```

三个注意点：

1. `pyproject.toml` 里 `server` 这一组可选依赖包含 vLLM 的 Linux wheel，Mac 上装不了。
   `--all-extras` 会把它拉进来。先试不带 extras 的 `uv sync --locked`，装不上再逐个排除。
   我们只需要客户端，不需要 vLLM。
2. `tufa-arc-agi-framework` 用的是相对路径 `../tufa-arc-agi-framework` 的可编辑安装，Son 包里那个目录在
   `src/` 下，相对位置正好对得上，不用改。
3. 依赖里的 `arc-agi-3-local`（包名 `re_arc`）来自 GitHub，装不上也不影响。
   只要在配置里显式给出 `environment.environments_dir`，`taaf` 就不会去 import `re_arc`。
   这一点在 `taaf/game_api.py` 的 `_resolve_environments_dir` 里写得很清楚：只有值为 `__auto__` 时才 import。
   装依赖需要联网时，`uv` 走 Clash 代理；跑实验时反过来必须 unset 代理。

装完先自检：`.venv/bin/python -c "import taaf, arcengine, arc_agi; print('ok')"`。

## 六、接 DeepSeek 的通道

首选 hub（`llm-model-hub-apis.sf-express.com`），理由是旧 harness 08-25 就是这么跑的，两边通道一致才谈得上对照。

要做的事：

1. 把 `hub-thinking-disabled.patch` 的 5 行移植进 Son 版的 `inference/utils/openai_compat.py`。
   移植后用 `diff` 逐字比对两个文件的这一段，确认一致再往下走。
2. 复制 `inference.hub-ds.json` 到 Son 版的 `configs/` 下，改 `experiments.root_dir` 指向
   `probes/20260903/harness_compare/`（目录已建好）。
3. 模型名 `aliyun/deepseek-v4-flash-0731` 是 08-25 记下的，hub 上的模型清单会变。
   先用一次最小请求确认当前可用名，别凭配置文件里的旧名开跑。
4. hub 的 429 是上游按「API Key × 模型」分桶的 TPM 限流，与 hub 自身的 QPM 无关。
   本地并发压到 1 到 3，别照搬线上的 28。

## 七、执行计划

每一阶段做完先看验收项，不过就停下来查，不要带着问题往下跑。

**阶段 0：环境自检**
装好 Son 版 venv，`import` 三件套通过。旧 harness 的 venv 已经好了，不要动它。

**阶段 1：单局冒烟**
两套 harness 各跑 1 局同一个局（建议 vc33，旧 harness 08-25 在这局上有三种子基线：2 关，58/172/129 步），
`n_passes=1`、`concurrent_jobs=1`、`max_runtime_minutes` 两边取同一个值（建议 40）。
验收：两边都产出完整 run 目录，都有实际动作发生，都没有整局零动作。
Son 版第一次跑很可能在 tool 解析、reasoning 解析或消息格式上报错，这些是 vLLM 侧参数在 hub 上不认导致的，逐个记录并修。

**阶段 2：同题对照正式跑**
局的选择：先用 5 局，覆盖不同难度。建议 vc33、ft09、ls20、r11l、sc25。
理由是这五局我们都有历史基线，r11l 是已知的稳定负样本，出现异常好判断。
每局每套 harness 跑 3 个种子，共 5 局 × 2 套 × 3 种子 = 30 路。
单种子结论一律不算，这是 08-24 定下的规矩（当时 R3/R4 两轮在单种子下当场反转）。
控制变量：同模型、同局、同种子数、同 `max_runtime_minutes`、同时间片、同并发、同关思考开关。
唯一变量是 harness 本身。开跑前把两边的实际请求体各抓一份存档，用来证明除 harness 外没有别的差异。

时间片这件事要特意确认：Son 版 `tool_agent.py` 有 `_yield_seconds`（第 1135 行读取，2241 行使用），
Kaggle 侧默认给 60 秒（`inference/framework/kaggle.py:114`）。旧 harness 的对应行为要查清楚，
两边不一致就显式设成一样，并在报告里写明设成了多少。

**阶段 3：出数**
统计口径两边必须用同一个脚本、同一份定义。至少给出逐局的：过关数、总步数、有效动作占比、
单局用时、模型调用次数、平均每次调用的输入 token。
不要只报平均分。08-01 的教训是同一批数据换三种判据能差 10 个百分点，所以判据要写在报告里。

**阶段 4：归因**
把阶段 3 的差距落到代码。从 `solver.py` 的 1843 行差异入手，参照解剖报告第五节已有的结论，
挑出 3 到 5 处机制，说明每处对哪些局有效、证据是哪几条日志。
最后按「搬过来的代价 / 预期收益」排序，给出可以进后续刀口的候选清单。

## 八、陷阱清单

都是实测踩过的，按踩到的概率排：

1. **代理方向是反的**。装依赖要走 Clash（`127.0.0.1:6789`），跑实验必须 `unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy`。
   走代理去请求 hub 会每轮读超时。
2. **不关思考会烧光预算**。DeepSeek 默认开思考，`thinking: {"type": "disabled"}` 必须带上。
3. **`max_tokens` 上限**。阿里通道上限 65536，给大了直接 400。旧配置里 `analyzer.max_output` 是 16000，照抄即可。
4. **`provider` 要填 `openrouter`**。填 `vllm` 会带上 `top_k` 之类 hub 不认的字段。
5. **429 是 TPM 限流**，不是并发数问题。降并发、加退避，别去调 QPM。
6. **`environments_dir` 必须显式给**，否则 `taaf` 会去 import 没装的 `re_arc`。
7. **`save_request_logs` 默认是关的**（两套配置里都是 `false`）。它管的是完整请求日志。
   另有一套 prompt 快照（`tool_agent.py` 里的 `_write_prompt_log_snapshot`）写到 `prompts/<局名>_p0.log`，每次调用覆盖写。
   第 1 刀依赖的是后者。开跑前现场确认这两条各自的开关状态，别凭本文断言。
8. **每局用时闸刀**。线上是 132 分钟一局，本地跑不起，两边统一取小值即可，但必须一致并写进报告。
9. **`timeout` 命令 Mac 上没有**，用循环代替。前台 `sleep` 会被拦，用 until 循环等条件。
10. **长任务要显进度**，收尾绝不用 `| tail -N`，那样只有 EOF 才出东西，看着像卡死。
11. **zsh 的通配符**。`--include=*.py` 这类参数要加引号。
12. **别动 `build_v12_curator/`**。那是周五要用的第 1 刀代码，本次对照只在 `reference/` 下和新建目录里做。

## 九、交付物

1. 本地 HTML 报告，路径 `notes/harness-compare-20260903.html`。
   样式参照 `notes/fnrep-v4-anatomy-20260903.html`。用户的报告铁律：多彩活泼，每页一屏不滚，不许省略。
   不要用 Artifact，产本地文件。
2. 原始跑数留在 `probes/20260903/harness_compare/`，不要清理。
3. 结论写进记忆：新建一篇 `arc3-harness-compare-20260903.md`，并在 `arc3-index.md` 加一行指向它。
   记忆正文里要有可反驳的数字，不要只写结论。
4. 如果某一项没跑成，在报告里明说没跑成和为什么，不要用别的数补位。

## 十、红线

- **不许提交 Kaggle**，不许花提交名额或 GPU 额度。任何提交动作由用户拍板。
- **不许开后台 agent，不许 fan out subagent**。
- **一天只上一个改动，不做兜底分支**。这次是纯对照实验，不改线上代码。
- **空结果先证明它是真的**，别把空当成结论。
- **说「一致」必须逐字比对**，凭印象不算。
- 删除走废纸篓，别用 rm。
- 报告不往桌面塞，进本项目的 `notes/`。

## 十一、可以直接抄的起手式

```bash
cd ~/Desktop/project/arc-agi-3/reference/fnrep-source-v4/src/ARC3-Inference
uv venv --python 3.12.12 .venv
env -u VIRTUAL_ENV uv sync --locked --python .venv/bin/python
.venv/bin/python -c "import taaf, arcengine, arc_agi; print('ok')"
```

跑一局的样板（照 `run_duck_vc33.sh` 改）：

```bash
set -a; source "$HOME/Desktop/2-快件质量与OpenClaw/langfuse-local/.env"; set +a
export OPENROUTER_API_KEY="$AIPLAT_KEY" OPERATION_MODE=OFFLINE HUB_THINKING_DISABLED=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
cd ~/Desktop/project/arc-agi-3/reference/fnrep-source-v4/src/ARC3-Inference
make interactive CONFIG_PATH=configs/inference.hub-ds.json GAME=vc33 \
  N_PASSES=1 MAX_RUNTIME_MINUTES=40 CONCURRENT_JOBS=1 RUN_NAME=fnv4-vc33-ds
```
