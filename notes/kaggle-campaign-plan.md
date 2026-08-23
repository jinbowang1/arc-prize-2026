# Kaggle 参赛作战计划 (建于 2026-08-21, 随进度更新)

目标: **终榜(11-02)前五**; 冲刺参照=Milestone 1 冠军 Duck 1.21%, 当前榜首 cstl 3.57%。
每天 1 次提交名额, 每一发都交当天手里最强的组合。

## 架构(已建成)

```
提交物 = Kaggle notebook + dataset(本仓 kaggle_agent/) + Kaggle Models 权重挂载
真提交: 网关(gateway:8001, 变量要自己硬编码) 打隐藏游戏, 每动作计分
本地开发: 同一套代码, 模型换公司 hub 的 DeepSeek 陪练, 游戏换公开 25 局
```

- `kaggle_agent/remote_env.py` — API-only 环境句柄(无克隆体, 每动作计分)
- `kaggle_agent/explorer.py` — v1 零模型新颖度探索(第一发, 管线验证用)
- `kaggle_agent/llm_agent.py` — v2 对话式(已被判据否掉, 留作对照)
- `kaggle_agent/repl_agent.py` — **v3/v4 主力**: 模型写 Python 代码观察/验证/行动;
  act() 唯一计分, verify_wm 强制世界模型验证
- `kaggle_agent/serve_vllm.py` — GPU notebook 里起 vLLM
- `kaggle_agent/notebook/` — 三个 notebook: CPU 版提交(v1)/GPU 冒烟/GPU 版提交(主力)
- `scripts/build_kaggle_bundle.py` — dataset+kernel 打包(含 NvidiaRtxPro6000)

## 关键判决(全部有实验钉着, 详见 notes/literature-update-20260821.md 与 git log)

1. **真提交没有克隆体** → harness 的 5/37 不能平移, LLM 归纳是唯一正路
2. **手 > 嘴**: 同模型同游戏, 对话式 150 步 0 关 vs REPL 5 步过关
3. **验证 > 世界模型本身**(Rodionov 消融): 不验证的世界模型比没有还差
4. **提示词一句话能毁掉一切**: "探索预算 20-40%"A/B 实锤是毒药, 已回退
5. **换模型 = 换书写习惯 = 新失效面**: Qwen 把数据贴进裸围栏, compile() 门禁修掉

## 当前状态 (08-23 收盘)

- 提交账: 第一发 explorer ERROR / 第二发 repl+qwen v3 **0.00** /
  **第三发 08-23 已交**(kernel v5: 复读断路器+官方采样参数+关思考+种类手册, PENDING)
- 关思考提速已上线: 27B 一轮从 60-90s 降到 ~10s, 每局调用 4-6 次 → ~30 次
- 🚨 复读定案(23c900a): 关思考+temp0.2 近贪婪 → r11l 23 轮逐字复读全灭;
  修=Qwen 官方 non-thinking 采样(0.7/0.8/presence 1.0)+复读断路器, GPU 彩排验证 84 轮零复读
- 🚨 A/B 判决(08-23, probes/20260823/): "行动要成段"指引**不合入**——
  条件性鼓励("方向明确后")在模型没读懂规则的局上从不触发, 反而抑制探索
  (r11l act 11→4, ls20 20→13); ⚠️同版本代码 ls20 彩排用满 100 动作 vs A/B 只走 20 步
  = **单局单 seed 方差大, 以后 A/B 至少 2 局×2 臂看方向一致才算数**
- 下一病灶: 行动密度走**协议层**不走提示词劝说——探索本身成段
  (probe_clicks 一键点一批连通块中心, 把 N 轮开口压成 1 轮), 或 explorer 喷洒
  攒 history + REPL 只管归纳的混合范式

## 路线图

- **v4(进行中)**: verify_wm 世界模型验证 + 简化步 已进系统提示;
  待做=verify_wm 失败条目自动转成"下一个实验建议"
- **v5**: 跨游戏技能库(OCM/Voyager 式, 验证过的 predict/工具函数沉淀复用)
- **持续**: 本地探针环路(DeepSeek 陪练+transcript 复盘)驱动迭代;
  强模型(V4-Pro/Kimi)测能力天花板, 27B 的差距=蒸馏空间
- **9-30 Milestone 2** 强制公开在即, 榜首方法到时见分晓

## 探针记录归档

`probes/20260821/` — 今天全部对照实验的完整对话记录(空回复破案/A/B 定责/
Qwen 围栏坑), 每个结论都可回溯到原始 transcript。
