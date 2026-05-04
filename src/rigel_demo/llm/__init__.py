"""Rigel LLM 基座能力。"""

from __future__ import annotations

from rigel_demo.config import (
    DEFAULT_CHAT_SYSTEM_PROMPT,
    DEFAULT_SUMMARY_SYSTEM_PROMPT,
    LLMConfig,
    LLMConfigSection,
    LLMConfigurationError,
)
from rigel_demo.llm.client import (
    LLMMessage,
    LangChainSummaryClient,
    build_langchain_chat_model,
    extract_message_text,
    normalize_messages,
    to_langchain_messages,
)

__all__ = [
    "DEFAULT_CHAT_SYSTEM_PROMPT",
    "DEFAULT_SUMMARY_SYSTEM_PROMPT",
    "LLMConfig",
    "LLMConfigSection",
    "LLMConfigurationError",
    "LLMMessage",
    "LangChainSummaryClient",
    "build_langchain_chat_model",
    "extract_message_text",
    "normalize_messages",
    "to_langchain_messages",
]
