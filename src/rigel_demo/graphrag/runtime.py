"""GraphRAG chat 的 FalkorDB 运行时连接。"""

from __future__ import annotations

import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rigel_demo.config import GraphRAGConfig


@dataclass(frozen=True, slots=True)
class EmbeddedFalkorDBRuntime:
    """LangChain 访问本地 FalkorDBLite 的运行时连接。"""

    client: Any
    host: str
    port: int


def build_falkordb_graph(*, config: GraphRAGConfig, graph_name: str) -> Any:
    from langchain_community.graphs import FalkorDBGraph

    return FalkorDBGraph(
        database=graph_name,
        host=config.host,
        port=config.port,
        username=config.username or "",
        password=config.password or "",
    )


def runtime_graphrag_config(
    config: GraphRAGConfig,
    embedded_database: EmbeddedFalkorDBRuntime | None,
) -> GraphRAGConfig:
    if embedded_database is None:
        return config
    return GraphRAGConfig(
        host=embedded_database.host,
        port=embedded_database.port,
        username=None,
        password=None,
    )


def start_embedded_falkordb_runtime(*, database_path: Path, host: str) -> EmbeddedFalkorDBRuntime:
    from redislite.falkordb_client import FalkorDB

    bind_host = embedded_bind_host(host)
    port = reserve_local_port(bind_host)
    client = FalkorDB(
        str(database_path),
        serverconfig={
            "bind": bind_host,
            "port": str(port),
        },
    )
    return EmbeddedFalkorDBRuntime(client=client, host=bind_host, port=port)


def embedded_bind_host(host: str) -> str:
    if host in {"127.0.0.1", "localhost"}:
        return "127.0.0.1"
    return "127.0.0.1"


def reserve_local_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((host, 0))
        return int(server_socket.getsockname()[1])
