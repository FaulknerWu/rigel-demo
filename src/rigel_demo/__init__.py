"""Rigel demo 原型包。"""

from __future__ import annotations

from typing import Any

from rigel_demo.core.graph_ir import (
    Anchor,
    EdgeType,
    Entity,
    File,
    GraphEdge,
    GraphIR,
    GraphNode,
    Module,
    NodeType,
    Repository,
    Summary,
)
from rigel_demo.storage.falkordb_store import FalkorDBConfig, FalkorDBStore


_CLI_EXPORTS = {
    "InitResult",
    "init_repository",
}
_INDEXER_EXPORTS = {
    "RepositoryIndexResult",
    "index_repository",
}
_JAVA_EXPORTS = {
    "JavaParseRequest",
    "JavaSemanticEdgeRequest",
    "enrich_java_semantic_edges",
    "parse_java_file",
}

__all__ = [
    "Anchor",
    "EdgeType",
    "Entity",
    "File",
    "FalkorDBConfig",
    "FalkorDBStore",
    "GraphEdge",
    "GraphIR",
    "GraphNode",
    "InitResult",
    "Module",
    "NodeType",
    "Repository",
    "RepositoryIndexResult",
    "Summary",
    "JavaParseRequest",
    "JavaSemanticEdgeRequest",
    "enrich_java_semantic_edges",
    "init_repository",
    "index_repository",
    "parse_java_file",
]


def __getattr__(name: str) -> Any:
    """按需加载可选能力，避免基础包导入提前绑定运行环境。"""

    if name in _CLI_EXPORTS:
        from rigel_demo import cli

        value = getattr(cli, name)
        globals()[name] = value
        return value

    if name in _INDEXER_EXPORTS:
        from rigel_demo.indexing import repository_indexer

        value = getattr(repository_indexer, name)
        globals()[name] = value
        return value

    if name not in _JAVA_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from rigel_demo import java

    value = getattr(java, name)
    globals()[name] = value
    return value
