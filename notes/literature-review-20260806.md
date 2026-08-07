# ARC-AGI-3 文献调研 (2026-08-06)

检索方式: WebSearch(arXiv/Semantic 面) + 逐篇 WebFetch arXiv 摘要页核验。以下 18 篇全部核过原文, 引用格式 arXiv:ID。

## 主题一: ARC 谱系与官方定调

- **On the Measure of Intelligence** (Chollet, arXiv:1911.01547, 2019)。智能 = 技能获取效率(skill-acquisition efficiency), 不是任务熟练度; ARC 的设计原则 = 只依赖 Core Knowledge 先验、排除语言和外部知识。整个赛系的哲学根基。
- **ARC Prize 2025: Technical Report** (Chollet, Knoop, Kamradt, Landers, arXiv:2601.10904, 2026-01)。ARC-AGI-2 赛季总结: 1455 队/15154 提交, 私榜最高 24%; 年度技术主题 = **refinement loops**(按任务迭代优化程序, 靠反馈信号引导, 含进化式程序合成); 零预训练 7M 小网络也能打出有竞争力的分; 预告 AGI-3 转向交互推理。
- **ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence** (ARC Prize Foundation, arXiv:2603.24621, 2026-03)。官方技术报告: 交互式 POMDP 环境考探索/规划/记忆/目标获取/对齐; 人类全通, 2026-03 时前沿模型全部 <1% (Gemini 3.1: 0.37% / GPT-5.4: 0.26% / Opus 4.6: 0.25% / Grok-4.20: 0%); 计分 = 对人类基准的效率(RHAE), 难度经大规模人测标定。

## 主题二: ARC-AGI-3 专门方法 — "可执行世界模型"路线全面胜出

按时间序, 能看到一条清晰的攻坚线:

- **Graph-Based Exploration** (Rudakov, Shock, Cowley, arXiv:2512.24156, Preview赛3rd)。零 LLM 纯图探索(帧哈希+frontier 穷举+连通域/状态栏掩蔽): 私榜 12/25 关, 修 bug 后中位 17。证明系统性探索本身值这么多分; 弱点=大状态空间不可解、状态栏变体会污染状态空间。
- **Executable World Models for ARC-AGI-3** (Rodionov, arXiv:2605.05138, 2026-05)。coding agent 维护可执行 Python 世界模型, 对历史观测**验证**, 靠重构**简化**(MDL 简洁性偏置), 在模型内**规划**后再行动; 严格禁止 per-game 硬编码。GPT-5.5-high: 15/25 游戏全通, mean RHAE 58.12。
- **Do Coding Agents Need Executable World Models, Simplification, and Verification?** (Rodionov, arXiv:2607.15439, 2026-07)。四层嵌套消融: ①**每个变体都随模型变强/推理预算变大而变好**(模型能力主导) ②可执行接口不必然赢纯文本(gpt-5.5 两档设置里文本反超灵活接口版) ③简化 4 组里 3 组有用 ④**完整验证在全部 4 组排第一**; gpt-5.6-sol + 验证变体公开集全通 ~99%。
- **OPINE-World** (Courtis, Li, Sanner, arXiv:2607.01531, 2026-07)。双 agent 分工: 行动 agent 只采集经验+选动作(读 replay buffer, 从不改模型), 合成 agent 只在出现反例时重写 `game_engine.py`(CEGIS 反例引导合成); 自然语言假设笔记做人类可读的中间层; **ontology error** = 对象类型 Dirichlet 后验熵 × 效应表行熵的 noisy-OR 组合, 高误差对象引导探索; 严格 exact-replay 验证+确定性双跑检查+静态分析防"读未来状态作弊"。Opus 4.8 双 agent: **20/25 游戏, 效率分 78.4**(基线 GPT-5.5: 14 游戏/63.8); WorldCoder 和神经隐世界模型基线 **0 游戏**。
- **Tycho** (Lehmann, Aioanei, Vahdati, arXiv:2607.28287, 2026-07-30, 开源 NIMI-research/Tycho 已 clone)。把环境形式化为 rendered deterministic Moore machines; 关键区分**决策帧/动画帧/终局帧**(同样的像素在不同帧类语义不同); 世界模型 = 自由形式 Python 程序(init_state/transition/render/outcome 四函数), 支持 ⊥ 弃权的部分预测; 四种编排策略对比, **actor 主动请求 builder 子代理**(delegation)最优: Opus 4.8 下 88.49 RHAE vs 无模型 79.07; 选定策略换强模型: **GPT-5.6-sol 与 Opus 5 都 25/25 游戏 183/183 关 RHAE 100.00**, Opus 5 总动作只有人类基准的 38.8%。两个反直觉发现: **transition 预测精度与游戏表现不相关**(Trigger 88.1% match 只得 83.07, Orchestrator 16.2% match 得 88.49——模拟得准≠识别出目标); 成本分析 $250-500/游戏 后边际收益递减。局限自述: 跨局改进靠人工分析轨迹, 无自动学习闭环; 早期极少能预测 game-over。

**路线结论**: 2026 年 3→7 月, 公开集从 <1% 打到 100 RHAE, 靠的都是同一族方法——LLM 写可执行 Python 世界模型 + 对 replay 严格验证 + 模型内规划。但全部依赖前沿 API 模型(Opus 4.8/5, GPT-5.5/5.6), 且 Rodionov 消融证明模型能力是第一因子。

## 主题三: 思想源头(这条路线从哪来)

- **EMPA / Theory-Based RL** (Tsividis, …, Tenenbaum, arXiv:2107.12544, 2021)。贝叶斯推断学"游戏引擎程序"作为生成模型, 90 个 Atari 风格游戏达到人类级学习效率(几分钟学会), 且复现人类探索轨迹的细粒度结构。ARC-AGI-3 官方设计和全部 SOTA 方法, 本质都是 EMPA 思想的 LLM 化。
- **WorldCoder** (Tang, Key, Ellis, arXiv:2402.12275, 2024)。LLM 通过交互写 Python 程序当世界模型, "解释观测 + 对奖励保持乐观"双目标; 比 deep RL 样本高效、比 ReAct 计算高效、代码可编辑迁移。是这条线的直接前驱——但注意 OPINE 实测它在 ARC-AGI-3 上 0 游戏, 说明"能写代码模型"距"能打交互游戏"还差探索与目标归纳两层。
- **Go-Explore** (Ecoffet, Huizinga, Lehman, Stanley, Clune, arXiv:1901.10995, 2019)。硬探索三原则: 记住到过的状态、先回到有希望的状态再探索、先在可控环境解题再模仿学习固化。Montezuma 4×SOTA。just-explore 的图探索是它的直系后代; "先回frontier再试新动作"正是 RHAE 省动作的关键结构。
- **Voyager** (Wang, …, Anandkumar, arXiv:2305.16291, 2023)。Minecraft 终身学习: 自动课程 + **可执行代码技能库**(可检索/可组合/防遗忘) + 迭代提示自验证。与用户的集体技能进化项目同构; 对 AGI-3 的启示 = 已验证的动作宏/子程序应沉淀成库跨关卡复用。

## 主题四: 旁证与边界

- **Reason to Play** (Csaba, …, Tenenbaum, Tomov 等, arXiv:2605.08019, 2026-05)。人玩新游戏时的行为+fMRI 对照: 前沿 LRM 比 model-free/model-based RL 和贝叶斯理论模型都更贴合人类行为与脑活动; **脑对齐反映的是模型对游戏状态的 in-context 表征, 而非下游规划**。→ 状态表征质量是根本。
- **BALROG** (Paglieri, …, Rocktäschel, arXiv:2411.13543, ICLR 2025)。LLM/VLM 游戏 agent 基准(含 NetHack): 简单游戏部分成功, 难游戏全线崩; **给视觉表征反而更差**——与 ARC 圈"文本序列化优于图像"的经验互证(Duck 的 segmentation 首选、原始网格不给, 同源)。
- **AGI Maze** (Potapov, arXiv:2607.00627, 2026-07)。小迷宫都解不动: vanilla LLM 在推理时无法内部维持迷宫表征, 用消息历史当工作记忆也不行。→ 状态必须外置(代码/文件/图), 不能指望模型脑内记。
- **WorldEvolver** (Zhang, Zhang, Ng, Deng, arXiv:2606.30639, 2026-06)。冻结参数, 只在部署时进化世界模型的上下文: 情景记忆(检索式模拟)+语义记忆(从预测-观测差提炼规则)+低置信预测过滤; ALFWorld/ScienceWorld 双提升。→ 测试时记忆修订这条便宜路线有效。
- (另: Qwen-AgentWorld 语言世界模型 arXiv:2606.24597 已在跟踪清单, 与本线互补。)

## 对参赛的推论 (最重要的一节)

1. **公开集已解, 比赛没解**。论文 100 RHAE 全靠前沿 API; Kaggle 断网只准开源权重, 榜首才 1.86%。**比赛的真命题 = 把"可执行世界模型+验证+规划"架构装进 27B 级开源模型**。这不是从零发明路线, 是移植+补模型短板。
2. **模型弱 → 架构要多扛**。Rodionov 消融说模型能力是第一因子, 那么开源 27B 上必须把确定性部分(验证、重放、规划、探索调度、帧分类)全部下沉到符号层代码, LLM 只干它唯一不可替代的事: **提出规则和目标假设**。这正好是我们 08-06 分层设计的方向, 现在有论文实证背书。
3. **直接可抄的部件清单**: Tycho 的决策帧/动画帧区分 + 四函数模型接口 + ⊥ 弃权; OPINE 的双 agent 分工(采集与合成分离防固化) + 反例触发合成 + 确定性双跑检查; Rodionov 的"完整验证第一"; just-explore 的 frontier 图探索当保底策略; Go-Explore 的"先回frontier"; Voyager 的已验证技能库。
4. **两个反直觉要牢记**: ①transition 模拟精度 ≠ 得分(目标识别更重要, Tycho 实证) ②可执行接口不必然赢文本(弱模型上文本可能更稳, Rodionov 实证)——harness 要 A/B 这两个选择, 别当定论。
5. **成本现实**: 论文烧 $250-500/游戏的前沿 token; Kaggle 上 27B 本地 vLLM 吞吐完全不同量级 → 每个 LLM 调用都要花在刀刃上, 符号层免费劳动最大化。
6. **开放缺口(也是论文机会)**: Tycho 自述"跨局改进靠人工、无自动学习闭环"; 私榜泛化没人验证过; 开源小模型上这套架构的表现是空白。做出来既是名次也是 Paper Track 素材。

## 本地资产
- `reference/duck-harness/`(M1 冠军+TAAF评测框架) · `reference/ARC-AGI-3-Agents/`(官方模板) · `reference/arc-agi-3-just-explore/`(图探索, MIT) · `reference/Tycho/`(公开集 SOTA, 2607.28287 官方实现)
