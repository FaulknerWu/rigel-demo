"""Java 解析与语义补全请求模型。"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MODULE_NAME = "root"
DEFAULT_MODULE_ECOSYSTEM = "maven"
DEFAULT_ZONE = "prod"
GENERATED_ZONE = "generated"
DEFAULT_LSP_TIMEOUT_SECONDS = 60
JAVA_ZONE_VALUES = {"prod", "test", "tooling", "vendor", GENERATED_ZONE}


@dataclass(frozen=True, slots=True)
class JavaParseRequest:
    """Java 文件解析请求。

    请求对象显式携带仓库和模块上下文，因为单文件源码本身无法可靠反推出
    所属模块、生态和运行分区。
    """

    repository_name: str
    module_name: str = DEFAULT_MODULE_NAME
    module_root_path: str = "."
    module_ecosystem: str = DEFAULT_MODULE_ECOSYSTEM
    zone: str = DEFAULT_ZONE
    file_zone: str = DEFAULT_ZONE

    def __post_init__(self) -> None:
        required_fields = (
            "repository_name",
            "module_name",
            "module_root_path",
            "module_ecosystem",
            "zone",
            "file_zone",
        )
        for field_name in required_fields:
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"JavaParseRequest.{field_name} 必须是非空字符串")
        for field_name in ("zone", "file_zone"):
            if getattr(self, field_name) not in JAVA_ZONE_VALUES:
                supported_values = "、".join(sorted(JAVA_ZONE_VALUES))
                raise ValueError(f"JavaParseRequest.{field_name} 仅支持：{supported_values}")


@dataclass(frozen=True, slots=True)
class JavaSemanticEdgeRequest:
    """Java 语义边补全请求。"""

    repository_root_path: str
    lsp_timeout_seconds: int = DEFAULT_LSP_TIMEOUT_SECONDS
