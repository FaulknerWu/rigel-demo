"""从环境变量加载 LLM 运行配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from dotenv import load_dotenv


ENV_FILE_NAME = ".env"
DEFAULT_GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_SYSTEM_PROMPT = (
    "你是 Rigel 的代码图谱分析助手。回答时优先基于用户给出的代码图谱、仓库上下文与当前问题，"
    "无法从上下文确认的内容要明确说明不确定。"
)


class LLMFormat(StrEnum):
    """LLM 请求格式。"""

    OPENAI_CHAT = "openai_chat"
    GOOGLE_GENERATE_CONTENT = "google_generate_content"
    OPENAI_RESPONSES = "openai_responses"


class LLMConfigurationError(ValueError):
    """LLM 配置不可用。"""


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """LLM 客户端所需的最小配置。"""

    provider: str
    format: LLMFormat
    model: str
    api_key: str
    base_url: str | None
    timeout_seconds: float
    system_prompt: str
    temperature: float | None = None
    max_output_tokens: int | None = None

    @classmethod
    def from_env(cls, repository_path: Path) -> "LLMConfig":
        """从目标仓库 `.env` 与当前进程环境读取 LLM 配置。"""

        env_path = repository_path / ENV_FILE_NAME
        if env_path.exists():
            load_dotenv(env_path, override=False)

        provider = _read_provider()
        llm_format = _read_format(provider)
        model = _require_env("RIGEL_LLM_MODEL")
        api_key = _read_api_key(provider)
        base_url = _read_base_url(provider, llm_format)
        timeout_seconds = _read_float("RIGEL_LLM_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)
        temperature = _read_optional_float("RIGEL_LLM_TEMPERATURE")
        max_output_tokens = _read_optional_int("RIGEL_LLM_MAX_OUTPUT_TOKENS")
        system_prompt = _read_env("RIGEL_LLM_SYSTEM_PROMPT") or DEFAULT_SYSTEM_PROMPT

        return cls(
            provider=provider,
            format=llm_format,
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            system_prompt=system_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )


def _read_provider() -> str:
    return (_read_env("RIGEL_LLM_PROVIDER") or "openai").lower()


def _read_format(provider: str) -> LLMFormat:
    format_value = (_read_env("RIGEL_LLM_FORMAT") or _default_format(provider).value).lower()
    try:
        return LLMFormat(format_value)
    except ValueError as error:
        supported_values = ", ".join(llm_format.value for llm_format in LLMFormat)
        raise LLMConfigurationError(f"RIGEL_LLM_FORMAT 仅支持：{supported_values}") from error


def _default_format(provider: str) -> LLMFormat:
    if provider == "google":
        return LLMFormat.GOOGLE_GENERATE_CONTENT
    return LLMFormat.OPENAI_RESPONSES


def _read_api_key(provider: str) -> str:
    api_key = _read_env("RIGEL_LLM_API_KEY")
    if api_key:
        return api_key

    provider_key_names = {
        "openai": ("OPENAI_API_KEY",),
        "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    }
    fallback_key_names = provider_key_names.get(provider, ())
    for key_name in fallback_key_names:
        value = _read_env(key_name)
        if value:
            return value

    fallback_names = ", ".join(("RIGEL_LLM_API_KEY", *fallback_key_names))
    raise LLMConfigurationError(f"缺少 LLM API Key，请配置 {fallback_names}")


def _read_base_url(provider: str, format: LLMFormat) -> str | None:
    custom_base_url = _read_env("RIGEL_LLM_BASE_URL")
    if custom_base_url:
        return custom_base_url

    if provider == "google" and format is LLMFormat.GOOGLE_GENERATE_CONTENT:
        return DEFAULT_GOOGLE_BASE_URL

    openai_base_url = _read_env("OPENAI_BASE_URL")
    if openai_base_url:
        return openai_base_url

    if provider != "openai":
        raise LLMConfigurationError("自定义提供商必须配置 RIGEL_LLM_BASE_URL")
    return None


def _require_env(name: str) -> str:
    value = _read_env(name)
    if value:
        return value
    raise LLMConfigurationError(f"缺少必要环境变量：{name}")


def _read_env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    stripped_value = value.strip()
    return stripped_value or None


def _read_float(name: str, default: float) -> float:
    value = _read_env(name)
    if value is None:
        return default
    try:
        parsed_value = float(value)
    except ValueError as error:
        raise LLMConfigurationError(f"{name} 必须是数字") from error
    if parsed_value <= 0:
        raise LLMConfigurationError(f"{name} 必须大于 0")
    return parsed_value


def _read_optional_float(name: str) -> float | None:
    value = _read_env(name)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError as error:
        raise LLMConfigurationError(f"{name} 必须是数字") from error


def _read_optional_int(name: str) -> int | None:
    value = _read_env(name)
    if value is None:
        return None
    try:
        parsed_value = int(value)
    except ValueError as error:
        raise LLMConfigurationError(f"{name} 必须是整数") from error
    if parsed_value <= 0:
        raise LLMConfigurationError(f"{name} 必须大于 0")
    return parsed_value
