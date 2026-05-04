"""基于 LangChain 的 ChatOpenAI 调用入口。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from rigel_demo.config import LLMConfig, LLMConfigurationError


MessageRole = Literal["user", "assistant"]


@dataclass(frozen=True, slots=True)
class LLMMessage:
    """发送给 LLM 的对话消息。"""

    role: MessageRole
    content: str


class LangChainSummaryClient:
    """使用 LangChain ChatOpenAI 生成检索摘要文本。"""

    def __init__(self, config: LLMConfig, *, chat_model: Any | None = None) -> None:
        self._config = config
        self._chat_model = chat_model or build_langchain_chat_model(config)

    @property
    def config(self) -> LLMConfig:
        return self._config

    def generate_reply(self, messages: Sequence[LLMMessage]) -> str:
        normalized_messages = normalize_messages(messages)
        if not normalized_messages:
            raise ValueError("消息列表不能为空")

        response = self._chat_model.invoke(to_langchain_messages(normalized_messages, system_prompt=self._config.system_prompt))
        response_text = extract_message_text(response)
        if not response_text:
            raise ValueError("LLM 未返回可展示文本")
        return response_text


def build_langchain_chat_model(config: LLMConfig) -> Any:
    """根据项目 LLM 配置构造 LangChain ChatOpenAI。"""

    try:
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = {
            "model": config.model,
            "api_key": config.api_key,
            "timeout": config.timeout_seconds,
        }
        if config.base_url is not None:
            kwargs["base_url"] = config.base_url
        if config.temperature is not None:
            kwargs["temperature"] = config.temperature
        if config.max_output_tokens is not None:
            kwargs["model_kwargs"] = {"max_completion_tokens": config.max_output_tokens}
        return ChatOpenAI(**kwargs)
    except Exception as error:
        raise LLMConfigurationError(f"LangChain ChatOpenAI 初始化失败：{error}") from error


def normalize_messages(messages: Sequence[LLMMessage]) -> list[LLMMessage]:
    """裁剪消息内容并拒绝空消息。"""

    normalized_messages: list[LLMMessage] = []
    for message in messages:
        content = message.content.strip()
        if not content:
            raise ValueError("消息内容不能为空")
        normalized_messages.append(LLMMessage(role=message.role, content=content))
    return normalized_messages


def to_langchain_messages(
    messages: Sequence[LLMMessage],
    *,
    system_prompt: str | None = None,
) -> list[tuple[str, str]]:
    """转换为 LangChain ChatModel 可直接接收的消息格式。"""

    langchain_messages: list[tuple[str, str]] = []
    if system_prompt:
        langchain_messages.append(("system", system_prompt))
    langchain_messages.extend((message.role, message.content) for message in messages)
    return langchain_messages


def extract_message_text(response: object) -> str:
    """从 LangChain AIMessage 或测试替身中提取文本。"""

    text_value = getattr(response, "text", None)
    if isinstance(text_value, str) and text_value.strip():
        return text_value.strip()

    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return _content_blocks_text(content)
    return ""


def _content_blocks_text(content_blocks: list[object]) -> str:
    texts: list[str] = []
    for content_block in content_blocks:
        if isinstance(content_block, str):
            texts.append(content_block)
        elif isinstance(content_block, dict) and isinstance(content_block.get("text"), str):
            texts.append(content_block["text"])
    return "\n".join(text.strip() for text in texts if text.strip()).strip()
