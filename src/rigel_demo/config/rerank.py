"""从仓库 `.rigel/config.json` 加载 Rerank 运行配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rigel_demo.config.document import ConfigDocumentErrorMessages, ConfigFieldReader, read_config_document

RERANK_CONFIG_SECTION_NAME = "rerank"


class RerankConfigurationError(ValueError):
    """Rerank 配置不可用。"""


@dataclass(frozen=True, slots=True)
class RerankConfig:
    """Rerank 客户端所需配置。"""

    provider: str
    base_url: str
    model: str
    api_key: str
    timeout_seconds: float
    top_n: int
    candidate_limit_per_query: int
    failover_enabled: bool = False

    @classmethod
    def from_repository(cls, repository_path: Path) -> "RerankConfig":
        """从目标仓库 `.rigel/config.json` 读取 Rerank 配置。"""

        reader = ConfigFieldReader(
            data=_read_rerank_config(repository_path),
            section=RERANK_CONFIG_SECTION_NAME,
            error_type=RerankConfigurationError,
        )
        provider = reader.required_string("provider").lower()
        if provider != "gitee_ai":
            raise RerankConfigurationError("rerank.provider 当前仅支持：gitee_ai")

        return cls(
            provider=provider,
            base_url=reader.required_string("base_url").rstrip("/"),
            model=reader.required_string("model"),
            api_key=reader.required_string("api_key"),
            timeout_seconds=reader.positive_float("timeout_seconds"),
            top_n=reader.positive_int("top_n"),
            candidate_limit_per_query=reader.positive_int("candidate_limit_per_query"),
            failover_enabled=_read_failover_enabled(reader),
        )


def _read_rerank_config(repository_path: Path) -> dict[str, Any]:
    config_document = read_config_document(
        repository_path,
        messages=ConfigDocumentErrorMessages(
            missing="缺少 Rerank 配置文件：{config_path}",
            read="读取 Rerank 配置文件失败：{config_path}",
            invalid_json="Rerank 配置文件不是合法 JSON：{config_path}",
            root="Rerank 配置文件根节点必须是 JSON 对象",
        ),
        error_type=RerankConfigurationError,
    )
    rerank_config = config_document.get(RERANK_CONFIG_SECTION_NAME)
    if not isinstance(rerank_config, dict):
        raise RerankConfigurationError("Rerank 配置文件必须包含对象字段：rerank")
    return rerank_config


def _read_failover_enabled(reader: ConfigFieldReader) -> bool:
    if "failover_enabled" not in reader.data:
        return False
    value = reader.data["failover_enabled"]
    if not isinstance(value, bool):
        raise RerankConfigurationError("rerank.failover_enabled 必须是布尔值")
    return value
