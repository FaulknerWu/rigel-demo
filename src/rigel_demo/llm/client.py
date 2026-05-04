"""基于 LangChain 的 ChatOpenAI 调用入口。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from langchain_openai import ChatOpenAI

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
        return ReasoningContentChatOpenAI(**kwargs)
    except Exception as error:
        raise LLMConfigurationError(f"LangChain ChatOpenAI 初始化失败：{error}") from error


class ReasoningContentChatOpenAI(ChatOpenAI):
    """保留 OpenAI-compatible thinking mode 的 reasoning_content。"""

    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        message_payloads = payload.get("messages")
        if not isinstance(message_payloads, list):
            return payload

        source_messages = self._convert_input(input_).to_messages()
        for source_message, message_payload in zip(source_messages, message_payloads, strict=False):
            if not isinstance(message_payload, dict) or message_payload.get("role") != "assistant":
                continue
            reasoning_content = _message_reasoning_content(source_message)
            if reasoning_content is not None:
                message_payload["reasoning_content"] = reasoning_content
        return payload

    def _create_chat_result(self, response: object, generation_info: dict[str, object] | None = None) -> Any:
        chat_result = super()._create_chat_result(response, generation_info=generation_info)
        response_document = _response_document(response)
        choices = response_document.get("choices")
        if not isinstance(choices, list):
            return chat_result

        for generation, choice in zip(chat_result.generations, choices, strict=False):
            if not isinstance(choice, Mapping):
                continue
            message = choice.get("message")
            if not isinstance(message, Mapping):
                continue
            reasoning_content = message.get("reasoning_content")
            if isinstance(reasoning_content, str):
                generation.message.additional_kwargs["reasoning_content"] = reasoning_content
        return chat_result


def _message_reasoning_content(message: object) -> str | None:
    reasoning_content = getattr(message, "reasoning_content", None)
    if isinstance(reasoning_content, str):
        return reasoning_content

    additional_kwargs = getattr(message, "additional_kwargs", None)
    if isinstance(additional_kwargs, Mapping):
        reasoning_content = additional_kwargs.get("reasoning_content")
        if isinstance(reasoning_content, str):
            return reasoning_content
    return None


def _response_document(response: object) -> Mapping[str, object]:
    if isinstance(response, Mapping):
        return response
    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        dumped_response = model_dump(exclude={"choices": {"__all__": {"message": {"parsed"}}}})
        if isinstance(dumped_response, Mapping):
            return dumped_response
    return {}


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
