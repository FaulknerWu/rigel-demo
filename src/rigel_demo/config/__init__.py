"""Rigel 仓库配置入口。"""

from __future__ import annotations

from rigel_demo.config.document import (
    RIGEL_CONFIG_FILE_NAME,
    RIGEL_WORKSPACE_DIRECTORY_NAME,
    ConfigDocumentErrorMessages,
    ConfigFieldReader,
    config_document_path,
    read_config_document,
)
from rigel_demo.config.embedding import (
    EmbeddingConfig,
    EmbeddingConfigurationError,
    EmbeddingFormat,
    EmbeddingInputMode,
)
from rigel_demo.config.graphrag import GraphRAGConfig, GraphRAGConfigurationError
from rigel_demo.config.llm import (
    DEFAULT_CHAT_SYSTEM_PROMPT,
    DEFAULT_SUMMARY_SYSTEM_PROMPT,
    LLMConfig,
    LLMConfigSection,
    LLMConfigurationError,
)
from rigel_demo.config.rerank import RerankConfig, RerankConfigurationError
from rigel_demo.config.web import DEFAULT_CONFIG_DOCUMENT, WebConfig, WebConfigurationError

__all__ = [
    "DEFAULT_CHAT_SYSTEM_PROMPT",
    "DEFAULT_CONFIG_DOCUMENT",
    "DEFAULT_SUMMARY_SYSTEM_PROMPT",
    "EmbeddingConfig",
    "EmbeddingConfigurationError",
    "EmbeddingFormat",
    "EmbeddingInputMode",
    "GraphRAGConfig",
    "GraphRAGConfigurationError",
    "LLMConfig",
    "LLMConfigSection",
    "LLMConfigurationError",
    "RerankConfig",
    "RerankConfigurationError",
    "RIGEL_CONFIG_FILE_NAME",
    "RIGEL_WORKSPACE_DIRECTORY_NAME",
    "ConfigDocumentErrorMessages",
    "ConfigFieldReader",
    "WebConfig",
    "WebConfigurationError",
    "config_document_path",
    "read_config_document",
]
