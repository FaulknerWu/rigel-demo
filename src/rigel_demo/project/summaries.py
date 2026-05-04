"""面向向量召回的 Summary 节点生成工具。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Protocol, cast

from rigel_demo.entities import Summary
from rigel_demo.graph.ir import EdgeType, GraphEdge, GraphIR, GraphNode, NodeType
from rigel_demo.embedding import RigelEmbedding
from rigel_demo.llm import LLMMessage, RigelLLM

RETRIEVAL_SUMMARY_PURPOSE = "retrieval"
SUMMARY_DESCRIBES_KIND = "retrieval-summary"
SUMMARY_SOURCE_HASH_PREFIX = "sha256:"

_SUMMARY_TARGET_TYPES = {NodeType.MODULE, NodeType.FILE, NodeType.ENTITY}


class SummaryClientConfig(Protocol):
    """Summary 依赖的模型配置契约。"""

    model: str


class SummaryEmbeddingClient(Protocol):
    """Summary 生成依赖的 Embedding 客户端接口。"""

    @property
    def config(self) -> SummaryClientConfig: ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


class SummaryTextClient(Protocol):
    """Summary 文本生成依赖的 LLM 客户端接口。"""

    @property
    def config(self) -> SummaryClientConfig: ...

    def generate_reply(self, messages: Sequence[LLMMessage]) -> str: ...


def attach_retrieval_summaries(
    graph: GraphIR,
    *,
    embedding_client: SummaryEmbeddingClient | RigelEmbedding,
    summary_client: SummaryTextClient | RigelLLM,
    target_node_ids: set[str] | None = None,
) -> GraphIR:
    """为可召回节点追加 Summary 节点和 DESCRIBES 边。"""

    existing_node_ids = {node.id for node in graph.nodes}
    existing_edge_ids = {edge.id for edge in graph.edges}
    target_nodes = [
        node
        for node in graph.nodes
        if node.type in _SUMMARY_TARGET_TYPES
        and (target_node_ids is None or node.id in target_node_ids)
    ]
    # 先生成全部摘要文本再批量 Embedding，减少外部模型调用次数并保持结果顺序可校验。
    summary_texts = [
        _generate_summary_text(target_node, summary_client=summary_client)
        for target_node in target_nodes
    ]
    embeddings = embedding_client.embed_texts(summary_texts)
    if len(embeddings) != len(target_nodes):
        raise ValueError("Embedding 返回数量与 Summary 目标数量不一致")

    embedding_model = embedding_client.config.model
    summary_model = summary_client.config.model
    for target_node, summary_text, embedding in zip(target_nodes, summary_texts, embeddings, strict=True):
        summary = build_retrieval_summary(
            target_node,
            text=summary_text,
            summary_model=summary_model,
            embedding_model=embedding_model,
            embedding=embedding,
        )
        if summary.summary_id not in existing_node_ids:
            graph.add_node(summary)
            existing_node_ids.add(summary.summary_id)

        describes_edge = GraphEdge.create(
            EdgeType.DESCRIBES,
            summary.summary_id,
            target_node.id,
            kind=SUMMARY_DESCRIBES_KIND,
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
    summary_model: str,
    embedding_model: str,
    embedding: list[float],
) -> Summary:
    """根据节点属性生成稳定的检索摘要。"""

    return Summary(
        summary_id=f"summary:{target_node.id}:retrieval",
        text=text,
        purpose=RETRIEVAL_SUMMARY_PURPOSE,
        source_hash=_source_hash(target_node, text),
        summary_model=summary_model,
        embedding_model=embedding_model,
        embedding_dimensions=len(embedding),
        embedding=embedding,
    )


def _generate_summary_text(target_node: GraphNode, *, summary_client: SummaryTextClient | RigelLLM) -> str:
    summary_text = _normalize_summary_text(
        summary_client.generate_reply([LLMMessage(role="user", content=_summary_prompt(target_node))])
    )
    if not summary_text:
        raise ValueError("Summary 模型返回空摘要")
    return summary_text


def _summary_prompt(target_node: GraphNode) -> str:
    properties_json = json.dumps(target_node.properties, ensure_ascii=False, sort_keys=True, indent=2)
    return (
        "请为以下代码图谱节点生成一条检索摘要。\n"
        "要求：摘要需要覆盖节点类型、名称、路径或限定名等关键信息；不要输出列表、Markdown 或解释。\n\n"
        f"节点类型：{target_node.type.value}\n"
        f"节点 ID：{target_node.id}\n"
        f"结构摘要：{_local_summary_text(target_node)}\n"
        f"节点属性：\n{properties_json}"
    )


def _normalize_summary_text(text: str) -> str:
    return " ".join(line.strip() for line in text.splitlines() if line.strip())


def _local_summary_text(target_node: GraphNode) -> str:
    properties = target_node.properties
    if target_node.type == NodeType.ENTITY:
        return _join_summary_parts(
            target_node.type.value,
            cast(str, properties["kind_norm"]),
            cast(str, properties["display_name"]),
            cast(str, properties["qualified_name"]),
            cast(str, properties["kind_raw"]),
        )
    if target_node.type == NodeType.FILE:
        return _join_summary_parts(
            target_node.type.value,
            cast(str, properties["language"]),
            cast(str, properties["relative_path"]),
            cast(str, properties["zone"]),
        )
    if target_node.type == NodeType.MODULE:
        return _join_summary_parts(
            target_node.type.value,
            cast(str, properties["name"]),
            cast(str, properties["root_path"]),
            cast(str, properties["ecosystem"]),
            cast(str, properties["zone"]),
        )
    return _join_summary_parts(target_node.type.value, target_node.id)


def _source_hash(target_node: GraphNode, text: str) -> str:
    for property_name in ("semantic_hash", "content_hash"):
        value = target_node.properties.get(property_name)
        if value:
            # Entity/File 已有内容哈希时直接继承，方便后续判断摘要是否跟随源码变化。
            return cast(str, value)

    source = f"{target_node.id}\n{text}".encode("utf-8")
    return f"{SUMMARY_SOURCE_HASH_PREFIX}{hashlib.sha256(source).hexdigest()}"


def _join_summary_parts(*parts: str) -> str:
    return " ".join(part for part in parts if part)
