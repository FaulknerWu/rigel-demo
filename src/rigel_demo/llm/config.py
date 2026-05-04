"""从仓库 `.rigel/config.json` 加载功能级 Chat Completions 运行配置。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from rigel_demo.config_document import ConfigDocumentErrorMessages, ConfigFieldReader, read_config_document

DEFAULT_CHAT_SYSTEM_PROMPT = (
    "你是 Rigel 的代码图谱分析助手。回答时优先基于用户给出的代码图谱、仓库上下文与当前问题，"
    "无法从上下文确认的内容要明确说明不确定。"
)
DEFAULT_SUMMARY_SYSTEM_PROMPT = (
    "你是 Rigel 的代码图谱摘要生成器。请基于给出的节点结构信息生成一条简洁、准确、"
    "适合语义检索的中文摘要，只输出摘要正文。"
)
MAX_SUMMARY_CONCURRENT_REQUESTS = 32


class LLMConfigSection(StrEnum):
    """`.rigel/config.json` 中的 LLM 功能配置段。"""

    CHAT = "chat"
    SUMMARY = "summary"


class LLMConfigurationError(ValueError):
    """LLM 配置不可用。"""


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """Chat Completions 客户端所需配置。"""

    provider: str
    model: str
    api_key: str
    base_url: str | None
    timeout_seconds: float
    system_prompt: str
    temperature: float | None = None
    max_output_tokens: int | None = None
    concurrent_requests: int = 1
    section: LLMConfigSection = LLMConfigSection.CHAT

    @classmethod
    def from_repository(
        cls,
        repository_path: Path,
        section: LLMConfigSection | str = LLMConfigSection.CHAT,
    ) -> "LLMConfig":
        """从目标仓库 `.rigel/config.json` 读取指定功能的 LLM 配置。"""

        normalized_section = _normalize_section(section)
        reader = ConfigFieldReader(
            data=_read_llm_config(repository_path, normalized_section),
            section=normalized_section.value,
            error_type=LLMConfigurationError,
        )
        provider = reader.required_string("provider").lower()

        return cls(
            provider=provider,
            model=reader.required_string("model"),
            api_key=reader.required_string("api_key"),
            base_url=_read_base_url(reader, provider),
            timeout_seconds=reader.positive_float("timeout_seconds"),
            system_prompt=reader.required_string("system_prompt"),
            temperature=reader.nullable_float("temperature"),
            max_output_tokens=reader.nullable_positive_int("max_output_tokens"),
            concurrent_requests=_read_concurrent_requests(reader, normalized_section),
            section=normalized_section,
        )


def _normalize_section(section: LLMConfigSection | str) -> LLMConfigSection:
    try:
        return LLMConfigSection(section)
    except ValueError as error:
        supported_values = ", ".join(config_section.value for config_section in LLMConfigSection)
        raise LLMConfigurationError(f"LLM 配置段仅支持：{supported_values}") from error


def _read_llm_config(repository_path: Path, section: LLMConfigSection) -> dict[str, Any]:
    config_document = read_config_document(
        repository_path,
        messages=ConfigDocumentErrorMessages(
            missing=f"缺少 {section.value} 配置文件：{{config_path}}",
            read=f"读取 {section.value} 配置文件失败：{{config_path}}",
            invalid_json=f"{section.value} 配置文件不是合法 JSON：{{config_path}}",
            root=f"{section.value} 配置文件根节点必须是 JSON 对象",
        ),
        error_type=LLMConfigurationError,
    )
    llm_config = config_document.get(section.value)
    if not isinstance(llm_config, dict):
        raise LLMConfigurationError(f"LLM 配置文件必须包含对象字段：{section.value}")
    return llm_config


def _read_base_url(reader: ConfigFieldReader, provider: str) -> str | None:
    custom_base_url = reader.nullable_string("base_url")
    if custom_base_url:
        return custom_base_url

    if provider != "openai":
        raise LLMConfigurationError(f"自定义 Chat Completions 提供商必须配置 {reader.field_path('base_url')}")
    return None


def _read_concurrent_requests(reader: ConfigFieldReader, section: LLMConfigSection) -> int:
    if section != LLMConfigSection.SUMMARY:
        return 1
    return reader.bounded_int("concurrent_requests", minimum=1, maximum=MAX_SUMMARY_CONCURRENT_REQUESTS)
