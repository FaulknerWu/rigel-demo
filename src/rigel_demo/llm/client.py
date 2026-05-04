"""基于 OpenAI SDK 的 Chat Completions 调用入口。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from openai import DefaultHttpxClient, OpenAI, OpenAIError

from rigel_demo.config import LLMConfig, LLMConfigurationError


MessageRole = Literal["user", "assistant"]


class LLMRequestError(RuntimeError):
    """LLM 远程请求失败。"""


class LLMResponseError(RuntimeError):
    """LLM 返回内容无法解析。"""


@dataclass(frozen=True, slots=True)
class LLMMessage:
    """发送给 LLM 的对话消息。"""

    role: MessageRole
    content: str


class RigelLLM:
    """封装 OpenAI Chat Completions 模型调用。"""

    def __init__(
        self,
        config: LLMConfig,
        *,
        openai_client: Any | None = None,
    ) -> None:
        self._config = config
        if openai_client is not None:
            # 测试或上层托管连接池时直接注入客户端，避免构造函数强依赖真实网络配置。
            self._client = openai_client
            return

        try:
            self._client = OpenAI(
                api_key=config.api_key,
                base_url=config.base_url,
                timeout=config.timeout_seconds,
                http_client=DefaultHttpxClient(trust_env=False),
            )
        except Exception as error:
            raise LLMConfigurationError(f"LLM 客户端初始化失败：{error}") from error

    @property
    def config(self) -> LLMConfig:
        """返回当前 LLM 配置。"""

        return self._config

    def generate_reply(self, messages: Sequence[LLMMessage]) -> str:
        """根据生成回复。"""

        normalized_messages = _normalize_messages(messages)
        if not normalized_messages:
            raise LLMResponseError("消息列表不能为空")

        try:
            return self._generate_with_chat(normalized_messages)
        except OpenAIError as error:
            raise LLMRequestError(f"LLM 调用失败：{error}") from error

    def _generate_with_chat(self, messages: Sequence[LLMMessage]) -> str:
        chat_messages: list[dict[str, str]] = []
        if self._config.system_prompt:
            chat_messages.append({"role": "system", "content": self._config.system_prompt})
        chat_messages.extend({"role": message.role, "content": message.content} for message in messages)

        request_body: dict[str, Any] = {
            "model": self._config.model,
            "messages": chat_messages,
        }
        _apply_optional_generation_options(request_body, self._config)

        completion = self._client.chat.completions.create(**request_body)
        for choice in completion.choices:
            content = choice.message.content
            if isinstance(content, str) and content.strip():
                return content.strip()
        raise LLMResponseError("LLM 未返回可展示文本")


def _normalize_messages(messages: Sequence[LLMMessage]) -> list[LLMMessage]:
    normalized_messages: list[LLMMessage] = []
    for message in messages:
        content = message.content.strip()
        if not content:
            raise LLMResponseError("消息内容不能为空")
        normalized_messages.append(LLMMessage(role=message.role, content=content))
    return normalized_messages


def _apply_optional_generation_options(
    request_body: dict[str, Any],
    config: LLMConfig,
) -> None:
    if config.temperature is not None:
        request_body["temperature"] = config.temperature
    if config.max_output_tokens is not None:
        request_body["max_completion_tokens"] = config.max_output_tokens
