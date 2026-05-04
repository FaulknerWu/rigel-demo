"""GraphRAG-SDK 连接配置。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rigel_demo.cli import RIGEL_CONFIG_FILE_NAME, RIGEL_WORKSPACE_DIRECTORY_NAME

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
        return cls(
            host=_require_string(section, "falkordb_host"),
            port=_require_port(section.get("falkordb_port")),
            username=_optional_string(section.get("falkordb_username"), "falkordb_username"),
            password=_optional_string(section.get("falkordb_password"), "falkordb_password"),
        )


def _read_config_document(repository_path: Path) -> dict[str, Any]:
    config_path = repository_path / RIGEL_WORKSPACE_DIRECTORY_NAME / RIGEL_CONFIG_FILE_NAME
    if not config_path.exists():
        raise FileNotFoundError(f"未找到配置文件，请先执行 rigel init 并填写配置：{config_path}")
    try:
        config_document = json.loads(config_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise GraphRAGConfigurationError(f"读取 GraphRAG 配置文件失败：{config_path}") from error
    except json.JSONDecodeError as error:
        raise GraphRAGConfigurationError(f"GraphRAG 配置文件不是合法 JSON：{config_path}") from error
    if not isinstance(config_document, dict):
        raise GraphRAGConfigurationError("GraphRAG 配置文件根节点必须是 JSON 对象")
    return config_document


def _require_string(section: dict[str, object], name: str) -> str:
    value = section.get(name)
    if not isinstance(value, str) or not value.strip():
        raise GraphRAGConfigurationError(f"graphrag.{name} 必须是非空字符串")
    return value.strip()


def _optional_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise GraphRAGConfigurationError(f"graphrag.{name} 必须是字符串或 null")
    return value.strip()


def _require_port(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GraphRAGConfigurationError("graphrag.falkordb_port 必须是整数")
    if value < 1 or value > 65_535:
        raise GraphRAGConfigurationError("graphrag.falkordb_port 必须在 1 到 65535 之间")
    return value
