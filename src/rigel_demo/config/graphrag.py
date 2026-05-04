"""GraphRAG-SDK 连接配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rigel_demo.config.document import ConfigDocumentErrorMessages, ConfigFieldReader, read_config_document

GRAPHRAG_CONFIG_SECTION_NAME = "graphrag"


class GraphRAGConfigurationError(ValueError):
    """GraphRAG-SDK 配置不可用。"""


@dataclass(frozen=True, slots=True)
class GraphRAGConfig:
    """GraphRAG-SDK 使用的 FalkorDB 服务连接配置。"""

    host: str
    port: int
    username: str | None
    password: str | None

    @classmethod
    def from_repository(cls, repository_path: Path) -> "GraphRAGConfig":
        config_document = _read_config_document(repository_path)
        section = config_document.get(GRAPHRAG_CONFIG_SECTION_NAME)
        if not isinstance(section, dict):
            raise GraphRAGConfigurationError(f"配置文件必须包含对象字段：{GRAPHRAG_CONFIG_SECTION_NAME}")
        reader = ConfigFieldReader(
            data=section,
            section=GRAPHRAG_CONFIG_SECTION_NAME,
            error_type=GraphRAGConfigurationError,
        )
        return cls(
            host=reader.required_string("falkordb_host"),
            port=reader.bounded_int("falkordb_port", minimum=1, maximum=65_535),
            username=reader.nullable_string("falkordb_username"),
            password=reader.nullable_string("falkordb_password"),
        )


def _read_config_document(repository_path: Path) -> dict[str, Any]:
    return read_config_document(
        repository_path,
        messages=ConfigDocumentErrorMessages(
            missing="未找到配置文件，请先执行 rigel init 并填写配置：{config_path}",
            read="读取 GraphRAG 配置文件失败：{config_path}",
            invalid_json="GraphRAG 配置文件不是合法 JSON：{config_path}",
            root="GraphRAG 配置文件根节点必须是 JSON 对象",
        ),
        missing_error_type=FileNotFoundError,
        error_type=GraphRAGConfigurationError,
    )
