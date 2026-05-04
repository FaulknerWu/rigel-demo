"""GraphRAG-SDK ontology 构建。"""

from __future__ import annotations

from typing import Any

from rigel_demo.config import GraphRAGConfig


def build_rigel_ontology(*, config: GraphRAGConfig, graph_name: str) -> Any:
    """从现有 Rigel FalkorDB 图谱提取 GraphRAG-SDK ontology。"""

    from falkordb import FalkorDB
    from graphrag_sdk.ontology import Ontology

    client = FalkorDB(
        host=config.host,
        port=config.port,
        username=config.username,
        password=config.password,
    )
    return Ontology.from_kg_graph(client.select_graph(graph_name))
