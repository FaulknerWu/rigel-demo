"""基于 OpenAI SDK 的 LLM 统一调用入口。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from openai import DefaultHttpxClient, OpenAI, OpenAIError

from rigel_demo.llm.config import LLMConfig, LLMConfigurationError, LLMFormat


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
    """统一封装不同提供商与请求格式。"""

    def __init__(
        self,
        config: LLMConfig,
        *,
        openai_client: Any | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._config = config
        self._http_client = http_client
        if openai_client is not None:
            # 测试或上层托管连接池时直接注入客户端，避免构造函数强依赖真实网络配置。
            self._client = openai_client
            return
        if config.format is LLMFormat.GOOGLE_GENERATE_CONTENT:
            # Google generateContent 与 OpenAI SDK 协议不同，这里用裸 HTTP 适配同一套消息模型。
            self._client = None
            self._http_client = http_client or httpx.Client(
                base_url=_require_base_url(config),
                timeout=config.timeout_seconds,
                trust_env=False,
            )
            return

        client_options: dict[str, Any] = {
            "api_key": config.api_key,
            "timeout": config.timeout_seconds,
            "http_client": DefaultHttpxClient(trust_env=False),
        }
        if config.base_url:
            client_options["base_url"] = config.base_url

        try:
            self._client = OpenAI(**client_options)
        except Exception as error:
            raise LLMConfigurationError(f"LLM 客户端初始化失败：{error}") from error

    @property
    def config(self) -> LLMConfig:
        """返回当前 LLM 配置。"""

        return self._config

    def generate_reply(self, messages: Sequence[LLMMessage]) -> str:
        """根据对话历史生成助手回复。"""

        normalized_messages = _normalize_messages(messages)
        if not normalized_messages:
            raise LLMResponseError("消息列表不能为空")

        try:
            if self._config.format is LLMFormat.OPENAI_RESPONSES:
                return self._generate_with_responses(normalized_messages)
            if self._config.format is LLMFormat.OPENAI_CHAT:
                return self._generate_with_chat(normalized_messages)
            return self._generate_with_google_generate_content(normalized_messages)
        except OpenAIError as error:
            raise LLMRequestError(f"LLM 调用失败：{error}") from error
        except httpx.HTTPError as error:
            raise LLMRequestError(f"LLM 调用失败：{error}") from error

    def _generate_with_responses(self, messages: Sequence[LLMMessage]) -> str:
        request_body: dict[str, Any] = {
            "model": self._config.model,
            "instructions": self._config.system_prompt,
            "input": [{"role": message.role, "content": message.content} for message in messages],
        }
        _apply_optional_generation_options(request_body, self._config, max_tokens_key="max_output_tokens")

        response = self._client.responses.create(**request_body)
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()
        raise LLMResponseError("LLM 未返回可展示文本")

    def _generate_with_chat(self, messages: Sequence[LLMMessage]) -> str:
        chat_messages: list[dict[str, str]] = []
        if self._config.system_prompt:
            # Chat Completions 没有 instructions 字段，系统提示必须作为首条 system 消息传入。
            chat_messages.append({"role": "system", "content": self._config.system_prompt})
        chat_messages.extend({"role": message.role, "content": message.content} for message in messages)

        request_body: dict[str, Any] = {
            "model": self._config.model,
            "messages": chat_messages,
        }
        _apply_optional_generation_options(request_body, self._config, max_tokens_key="max_tokens")

        completion = self._client.chat.completions.create(**request_body)
        for choice in completion.choices:
            content = getattr(choice.message, "content", None)
            if isinstance(content, str) and content.strip():
                return content.strip()
        raise LLMResponseError("LLM 未返回可展示文本")

    def _generate_with_google_generate_content(self, messages: Sequence[LLMMessage]) -> str:
        request_body: dict[str, Any] = {
            "contents": [_to_google_content(message) for message in messages],
        }
        if self._config.system_prompt:
            request_body["system_instruction"] = {
                "parts": [{"text": self._config.system_prompt}],
            }

        generation_config = _google_generation_config(self._config)
        if generation_config:
            request_body["generationConfig"] = generation_config

        response = _require_http_client(self._http_client).post(
            _google_generate_content_path(self._config.model),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self._config.api_key,
            },
            json=request_body,
        )
        response.raise_for_status()

        output_text = _extract_google_output_text(response.json())
        if output_text:
            return output_text
        raise LLMResponseError("LLM 未返回可展示文本")


def _normalize_messages(messages: Sequence[LLMMessage]) -> list[LLMMessage]:
    normalized_messages: list[LLMMessage] = []
    for message in messages:
        content = message.content.strip()
        if content:
            normalized_messages.append(LLMMessage(role=message.role, content=content))
    return normalized_messages


def _apply_optional_generation_options(
    request_body: dict[str, Any],
    config: LLMConfig,
    *,
    max_tokens_key: str,
) -> None:
    # Responses、Chat Completions 与 Google API 的 token 字段名不同，由调用方显式传入。
    if config.temperature is not None:
        request_body["temperature"] = config.temperature
    if config.max_output_tokens is not None:
        request_body[max_tokens_key] = config.max_output_tokens


def _google_generation_config(config: LLMConfig) -> dict[str, object]:
    generation_config: dict[str, object] = {}
    if config.temperature is not None:
        generation_config["temperature"] = config.temperature
    if config.max_output_tokens is not None:
        generation_config["maxOutputTokens"] = config.max_output_tokens
    return generation_config


def _to_google_content(message: LLMMessage) -> dict[str, object]:
    # Google API 使用 model 表示助手历史，进入本模块前仍统一暴露 assistant 角色。
    role = "model" if message.role == "assistant" else "user"
    return {
        "role": role,
        "parts": [{"text": message.content}],
    }


def _google_generate_content_path(model: str) -> str:
    if model.startswith(("models/", "tunedModels/")):
        return f"/{model}:generateContent"
    return f"/models/{model}:generateContent"


def _extract_google_output_text(response_body: dict[str, Any]) -> str:
    text_parts: list[str] = []
    candidates = response_body.get("candidates")
    if not isinstance(candidates, list):
        return ""

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        if not isinstance(content, dict):
            continue
        parts = content.get("parts")
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                text_parts.append(text.strip())
    return "\n".join(text_parts).strip()


def _require_base_url(config: LLMConfig) -> str:
    if config.base_url:
        return config.base_url
    raise LLMConfigurationError("当前请求格式必须配置 llm.base_url")


def _require_http_client(http_client: httpx.Client | None) -> httpx.Client:
    if http_client is None:
        raise LLMConfigurationError("Google 请求客户端未初始化")
    return http_client
