"""推理客户端层: OpenAI-compatible 后端 + Mock 后端。

设计约束:
- 零第三方依赖(urllib 直连), vLLM serve 原生就是 /v1/chat/completions。
- 换模型 = 换 base_url/model 两个配置, agent 代码一行不动。
- 本地 Mac 无 GPU: 循环逻辑的测试全走 MockLLM(脚本化应答), 不烧真推理。
- 记账: token 用量与延迟都记, 提交前要能回答"一局烧多少推理"。
"""
from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass, field


@dataclass
class LLMStats:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    errors: int = 0


class LLMClient:
    """OpenAI-compatible chat 客户端(vLLM serve / 任何兼容网关)。"""

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY",
                 timeout: float = 300.0, max_tokens: int = 6000, temperature: float = 0.2):
        # ⚠️max_tokens 是"思考+正文"的总预算: 推理型模型(deepseek 等)上下文
        # 一长思考就变长, 1600 实测被思考吃光 -> content 为空 -> 轮轮解析失败。
        # 默认给足 6000, 别省这个钱(动作比 token 贵)。
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.stats = LLMStats()

    def chat(self, messages: list[dict], **kw) -> str:
        """一次对话补全。失败重试 2 次后抛出(上层决定降级)。"""
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": kw.get("max_tokens", self.max_tokens),
            "temperature": kw.get("temperature", self.temperature),
        }
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
        )
        last: Exception | None = None
        for _ in range(3):
            t0 = time.monotonic()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.loads(r.read())
                self.stats.calls += 1
                self.stats.seconds += time.monotonic() - t0
                u = data.get("usage") or {}
                self.stats.prompt_tokens += u.get("prompt_tokens", 0)
                self.stats.completion_tokens += u.get("completion_tokens", 0)
                msg = data["choices"][0]["message"]
                content = msg.get("content") or ""
                if not content.strip():
                    # 思考吃光预算时 content 为空; 思考尾部常已含答案, 捞一把
                    content = (msg.get("reasoning_content") or "")[-2000:]
                return content
            except Exception as e:  # noqa: BLE001
                last = e
                self.stats.errors += 1
                time.sleep(2)
        raise RuntimeError(f"LLM 调用三次失败: {last!r}")


class MockLLM:
    """脚本化应答, 供本地无 GPU 测试主循环逻辑。

    responses 耗尽后循环复用最后一条(让长循环测试不至于崩)。
    """

    def __init__(self, responses: list[str]):
        assert responses, "MockLLM 至少要一条应答"
        self._responses = list(responses)
        self._i = 0
        self.stats = LLMStats()
        self.prompts: list[list[dict]] = []  # 测试断言用: 真实收到的 prompt

    def chat(self, messages: list[dict], **kw) -> str:
        self.prompts.append(messages)
        self.stats.calls += 1
        r = self._responses[min(self._i, len(self._responses) - 1)]
        self._i += 1
        return r


def parse_json_block(text: str) -> dict:
    """从模型输出里抠 JSON(容忍 ```json 围栏与前后闲话)。解析不出返回 {}。

    ⚠️宽容是刻意的: 主循环必须假设模型输出随时不合法, 解析失败=跳过本轮
    建议退回探索, 绝不让一次坏输出崩掉整局。
    """
    s = text
    if "```" in s:
        for part in s.split("```"):
            p = part.strip()
            if p.startswith("json"):
                p = p[4:]
            p = p.strip()
            if p.startswith("{"):
                s = p
                break
    start = s.find("{")
    if start < 0:
        return {}
    depth = 0
    for i, ch in enumerate(s[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[start:i + 1])
                except json.JSONDecodeError:
                    return {}
    return {}
