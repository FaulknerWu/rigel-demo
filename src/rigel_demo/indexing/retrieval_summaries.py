"""面向向量召回的 Summary 节点生成工具。"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping
from typing import Protocol

from rigel_demo.core.graph_ir import EdgeType, GraphEdge, GraphIR, GraphNode, NodeType, Summary
from rigel_demo.embedding import RigelEmbedding

RETRIEVAL_SUMMARY_PURPOSE = "retrieval"
SUMMARY_DESCRIBES_KIND = "retrieval-summary"
SUMMARY_PROVENANCE = "embedding"
SUMMARY_SOURCE_HASH_PREFIX = "sha256:"

_SUMMARY_TARGET_TYPES = {NodeType.MODULE, NodeType.FILE, NodeType.ENTITY}


class SummaryEmbeddingClient(Protocol):
    """Summary 生成依赖的最小 Embedding 客户端接口。"""

    @property
    def config(self) -> object: ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


def attach_retrieval_summaries(graph: GraphIR, *, embedding_client: SummaryEmbeddingClient | RigelEmbedding) -> GraphIR:
    """为可召回节点追加 Summary 节点和 DESCRIBES 边。"""

    existing_node_ids = {node.id for node in graph.nodes}
    existing_edge_ids = {edge.id for edge in graph.edges}
    target_nodes = [
        node
        for node in graph.nodes
        if node.type in _SUMMARY_TARGET_TYPES
    ]
    summary_texts = [_summary_text(target_node) for target_node in target_nodes]
    embeddings = embedding_client.embed_texts(summary_texts)
    if len(embeddings) != len(target_nodes):
        raise ValueError("Embedding 返回数量与 Summary 目标数量不一致")

    embedding_model = str(getattr(embedding_client.config, "model"))
    for target_node, summary_text, embedding in zip(target_nodes, summary_texts, embeddings, strict=True):
        summary = build_retrieval_summary(target_node, text=summary_text, embedding_model=embedding_model, embedding=embedding)
        if summary.summary_id not in existing_node_ids:
            graph.add_node(summary)
            existing_node_ids.add(summary.summary_id)

        describes_edge = GraphEdge.create(
            EdgeType.DESCRIBES,
            summary.summary_id,
            target_node.id,
            kind=SUMMARY_DESCRIBES_KIND,
            provenance=SUMMARY_PROVENANCE,
            confidence=1.0,
        )
        if describes_edge.id not in existing_edge_ids:
            graph.add_edge(describes_edge)
            existing_edge_ids.add(describes_edge.id)

    return graph


def build_retrieval_summary(
    target_node: GraphNode,
    *,
    text: str,
    embedding_model: str,
    embedding: list[float],
) -> Summary:
    """根据节点属性生成稳定的检索摘要。"""

    return Summary(
        summary_id=f"summary:{target_node.id}:retrieval",
        text=text,
        purpose=RETRIEVAL_SUMMARY_PURPOSE,
        source_hash=_source_hash(target_node, text),
        embedding_model=embedding_model,
        embedding_dimensions=len(embedding),
        embedding=embedding,
    )


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    """计算两个向量的余弦相似度。"""

    left_values = list(left)
    right_values = list(right)
    if not left_values or not right_values or len(left_values) != len(right_values):
        return 0.0

    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(left * right for left, right in zip(left_values, right_values, strict=True)) / (left_norm * right_norm)


def _summary_text(target_node: GraphNode) -> str:
    properties = target_node.properties
    if target_node.type == NodeType.ENTITY:
        return _join_summary_parts(
            target_node.type.value,
            _read_string(properties, "kind_norm"),
            _read_string(properties, "display_name"),
            _read_string(properties, "qualified_name"),
            _read_string(properties, "kind_raw"),
        )
    if target_node.type == NodeType.FILE:
        return _join_summary_parts(
            target_node.type.value,
            _read_string(properties, "language"),
            _read_string(properties, "relative_path"),
            _read_string(properties, "zone"),
        )
    if target_node.type == NodeType.MODULE:
        return _join_summary_parts(
            target_node.type.value,
            _read_string(properties, "name"),
            _read_string(properties, "root_path"),
            _read_string(properties, "ecosystem"),
            _read_string(properties, "zone"),
        )
    return _join_summary_parts(target_node.type.value, target_node.id)


def _source_hash(target_node: GraphNode, text: str) -> str:
    for property_name in ("semantic_hash", "content_hash"):
        value = target_node.properties.get(property_name)
        if isinstance(value, str) and value:
            return value

    source = f"{target_node.id}\n{text}".encode("utf-8")
    return f"{SUMMARY_SOURCE_HASH_PREFIX}{hashlib.sha256(source).hexdigest()}"


def _join_summary_parts(*parts: str) -> str:
    return " ".join(part for part in parts if part)


def _read_string(properties: Mapping[str, object], name: str) -> str:
    value = properties.get(name)
    return value if isinstance(value, str) else ""
