"""面向向量召回的 Summary 节点生成工具。"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Literal, Protocol, cast

from rigel_demo.entities import Summary
from rigel_demo.graph.ir import EdgeType, GraphEdge, GraphIR, GraphNode, NodeType
from rigel_demo.embedding import RigelEmbedding
from rigel_demo.llm import LLMMessage
from rigel_demo.prompts import build_summary_prompt

RETRIEVAL_SUMMARY_PURPOSE = "retrieval"
SUMMARY_DESCRIBES_KIND = "retrieval-summary"
SUMMARY_SOURCE_HASH_PREFIX = "sha256:"

_SUMMARY_TARGET_TYPES = {NodeType.MODULE, NodeType.FILE, NodeType.ENTITY}
SummaryProgressStage = Literal["summary_text", "summary_embedding"]


@dataclass(frozen=True, slots=True)
class SummaryProgress:
    """检索摘要构建的进度事件。"""

    stage: SummaryProgressStage
    current: int
    total: int
    detail: str


SummaryProgressReporter = Callable[[SummaryProgress], None]


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
    summary_client: SummaryTextClient,
    target_node_ids: set[str] | None = None,
    progress_reporter: SummaryProgressReporter | None = None,
) -> GraphIR:
    """为可召回节点追加 Summary 节点和 DESCRIBES 边。"""

    summary_model = summary_client.config.model
    embedding_model = embedding_client.config.model
    embedding_dimensions = _embedding_dimensions_from_config(embedding_client.config)
    _remove_stale_retrieval_summaries(
        graph,
        target_node_ids=target_node_ids,
        summary_model=summary_model,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
    )
    existing_node_ids = {node.id for node in graph.nodes}
    existing_edge_ids = {edge.id for edge in graph.edges}
    target_nodes = [
        node
        for node in graph.nodes
        if node.type in _SUMMARY_TARGET_TYPES
        and (target_node_ids is None or node.id in target_node_ids)
        and _summary_id(node) not in existing_node_ids
    ]
    _attach_existing_summary_edges(graph, existing_node_ids, existing_edge_ids, target_node_ids=target_node_ids)

    # 先生成全部摘要文本再批量 Embedding，减少外部模型调用次数并保持结果顺序可校验。
    if not target_nodes:
        return graph
    summary_texts = _generate_summary_texts(
        target_nodes,
        summary_client=summary_client,
        progress_reporter=progress_reporter,
    )
    embeddings = _generate_summary_embeddings(
        summary_texts,
        embedding_client=embedding_client,
        progress_reporter=progress_reporter,
    )
    if len(embeddings) != len(target_nodes):
        raise ValueError("Embedding 返回数量与 Summary 目标数量不一致")

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


def _remove_stale_retrieval_summaries(
    graph: GraphIR,
    *,
    target_node_ids: set[str] | None,
    summary_model: str,
    embedding_model: str,
    embedding_dimensions: int | None,
) -> None:
    summary_nodes_by_id = {
        node.id: node
        for node in graph.nodes
        if node.type == NodeType.SUMMARY
    }
    stale_summary_ids = {
        _summary_id(target_node)
        for target_node in graph.nodes
        if target_node.type in _SUMMARY_TARGET_TYPES
        and (target_node_ids is None or target_node.id in target_node_ids)
        and not _has_current_summary_node(
            summary_nodes_by_id,
            target_node,
            summary_model=summary_model,
            embedding_model=embedding_model,
            embedding_dimensions=embedding_dimensions,
        )
    }
    if not stale_summary_ids:
        return

    graph.nodes = [node for node in graph.nodes if node.id not in stale_summary_ids]
    graph.edges = [
        edge
        for edge in graph.edges
        if edge.source_id not in stale_summary_ids and edge.target_id not in stale_summary_ids
    ]


def _has_current_summary_node(
    summary_nodes_by_id: dict[str, GraphNode],
    target_node: GraphNode,
    *,
    summary_model: str,
    embedding_model: str,
    embedding_dimensions: int | None,
) -> bool:
    summary_node = summary_nodes_by_id.get(_summary_id(target_node))
    if summary_node is None:
        return False
    return _summary_matches_target(
        summary_node,
        target_node,
        summary_model=summary_model,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
    )


def _summary_matches_target(
    summary_node: GraphNode,
    target_node: GraphNode,
    *,
    summary_model: str,
    embedding_model: str,
    embedding_dimensions: int | None,
) -> bool:
    properties = summary_node.properties
    target_source_hash = _target_source_hash(target_node)
    if target_source_hash is not None and properties.get("source_hash") != target_source_hash:
        return False
    if embedding_dimensions is not None and properties.get("embedding_dimensions") != embedding_dimensions:
        return False
    return (
        properties.get("purpose") == RETRIEVAL_SUMMARY_PURPOSE
        and properties.get("summary_model") == summary_model
        and properties.get("embedding_model") == embedding_model
        and properties.get("embedding_dimensions") == _summary_embedding_dimensions(summary_node)
    )


def _attach_existing_summary_edges(
    graph: GraphIR,
    existing_node_ids: set[str],
    existing_edge_ids: set[str],
    *,
    target_node_ids: set[str] | None,
) -> None:
    for target_node in graph.nodes:
        if target_node.type not in _SUMMARY_TARGET_TYPES:
            continue
        if target_node_ids is not None and target_node.id not in target_node_ids:
            continue
        summary_id = _summary_id(target_node)
        if summary_id not in existing_node_ids:
            continue
        describes_edge = GraphEdge.create(
            EdgeType.DESCRIBES,
            summary_id,
            target_node.id,
            kind=SUMMARY_DESCRIBES_KIND,
            confidence=1.0,
        )
        if describes_edge.id not in existing_edge_ids:
            graph.add_edge(describes_edge)
            existing_edge_ids.add(describes_edge.id)


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
        summary_id=_summary_id(target_node),
        text=text,
        purpose=RETRIEVAL_SUMMARY_PURPOSE,
        source_hash=_source_hash(target_node, text),
        summary_model=summary_model,
        embedding_model=embedding_model,
        embedding_dimensions=len(embedding),
        embedding=embedding,
    )


def _summary_id(target_node: GraphNode) -> str:
    return f"summary:{target_node.id}:retrieval"


def _generate_summary_texts(
    target_nodes: list[GraphNode],
    *,
    summary_client: SummaryTextClient,
    progress_reporter: SummaryProgressReporter | None,
) -> list[str]:
    summary_texts = [""] * len(target_nodes)
    total = len(target_nodes)
    completed_count = 0
    max_workers = _summary_concurrent_requests(summary_client.config, total)
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="rigel-summary") as executor:
        future_indexes = {
            executor.submit(_generate_summary_text, target_node, summary_client=summary_client): index
            for index, target_node in enumerate(target_nodes)
        }
        for future in as_completed(future_indexes):
            index = future_indexes[future]
            summary_texts[index] = future.result()
            completed_count += 1
            if progress_reporter is not None:
                progress_reporter(
                    SummaryProgress(
                        stage="summary_text",
                        current=completed_count,
                        total=total,
                        detail=_summary_progress_detail(target_nodes[index]),
                    )
                )
    return summary_texts


def _generate_summary_text(target_node: GraphNode, *, summary_client: SummaryTextClient) -> str:
    summary_text = _normalize_summary_text(
        summary_client.generate_reply([LLMMessage(role="user", content=build_summary_prompt(target_node))])
    )
    if not summary_text:
        raise ValueError("Summary 模型返回空摘要")
    return summary_text


def _generate_summary_embeddings(
    summary_texts: list[str],
    *,
    embedding_client: SummaryEmbeddingClient | RigelEmbedding,
    progress_reporter: SummaryProgressReporter | None,
) -> list[list[float]]:
    embeddings: list[list[float]] = []
    total = len(summary_texts)
    batch_size = _embedding_batch_size(embedding_client.config, total)
    for start_index in range(0, total, batch_size):
        end_index = min(start_index + batch_size, total)
        if progress_reporter is not None:
            progress_reporter(
                SummaryProgress(
                    stage="summary_embedding",
                    current=start_index + 1,
                    total=total,
                    detail=f"{start_index + 1}-{end_index}",
                )
            )
        batch_embeddings = embedding_client.embed_texts(summary_texts[start_index:end_index])
        if len(batch_embeddings) != end_index - start_index:
            raise ValueError("Embedding 返回数量与 Summary 批次数量不一致")
        embeddings.extend(batch_embeddings)
    return embeddings


def _embedding_batch_size(config: SummaryClientConfig, total: int) -> int:
    if getattr(config, "input_mode", None) == "string":
        return 1
    batch_size = getattr(config, "batch_size", total)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        return total
    return batch_size


def _summary_concurrent_requests(config: SummaryClientConfig, total: int) -> int:
    concurrent_requests = getattr(config, "concurrent_requests", 1)
    if isinstance(concurrent_requests, bool) or not isinstance(concurrent_requests, int) or concurrent_requests <= 0:
        return 1
    return min(concurrent_requests, total)


def _summary_progress_detail(target_node: GraphNode) -> str:
    properties = target_node.properties
    if target_node.type == NodeType.FILE:
        return str(properties.get("relative_path", target_node.id))
    if target_node.type == NodeType.ENTITY:
        return str(properties.get("qualified_name") or properties.get("display_name") or target_node.id)
    if target_node.type == NodeType.MODULE:
        return str(properties.get("name", target_node.id))
    return target_node.id


def _normalize_summary_text(text: str) -> str:
    return " ".join(line.strip() for line in text.splitlines() if line.strip())


def _source_hash(target_node: GraphNode, text: str) -> str:
    target_source_hash = _target_source_hash(target_node)
    if target_source_hash is not None:
        return target_source_hash

    source = f"{target_node.id}\n{text}".encode("utf-8")
    return f"{SUMMARY_SOURCE_HASH_PREFIX}{hashlib.sha256(source).hexdigest()}"


def _target_source_hash(target_node: GraphNode) -> str | None:
    for property_name in ("semantic_hash", "content_hash"):
        value = target_node.properties.get(property_name)
        if value:
            # Entity/File 已有内容哈希时直接继承，方便后续判断摘要是否跟随源码变化。
            return cast(str, value)
    return None


def _summary_embedding_dimensions(summary_node: GraphNode) -> int | None:
    value = summary_node.properties.get("embedding")
    if isinstance(value, list):
        return len(value)
    return None


def _embedding_dimensions_from_config(config: SummaryClientConfig) -> int | None:
    value = getattr(config, "dimensions", None)
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None
