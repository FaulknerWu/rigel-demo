"""Java 解析与语义补全请求模型。"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MODULE_NAME = "root"
DEFAULT_MODULE_ECOSYSTEM = "maven"
DEFAULT_ZONE = "prod"


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


@dataclass(frozen=True, slots=True)
class JavaSemanticEdgeRequest:
    """Java 语义边补全请求。"""

    repository_root_path: str
    lsp_timeout_seconds: int = 30
