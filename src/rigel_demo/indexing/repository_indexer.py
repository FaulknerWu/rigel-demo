"""仓库级代码图谱索引流程。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rigel_demo.core.graph_ir import EdgeType, GraphEdge, GraphIR, Module, Repository
from rigel_demo.embedding import EmbeddingConfig, RigelEmbedding
from rigel_demo.indexing.retrieval_summaries import (
    SummaryEmbeddingClient,
    SummaryTextClient,
    attach_retrieval_summaries,
)
from rigel_demo.java import JavaParseRequest, JavaSemanticEdgeRequest, enrich_java_semantic_edges, parse_java_file
from rigel_demo.java.requests import DEFAULT_MODULE_ECOSYSTEM, DEFAULT_MODULE_NAME, DEFAULT_ZONE
from rigel_demo.llm import LLMConfig, LLMConfigSection, RigelLLM


IGNORED_DIRECTORY_NAMES = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".rigel",
    ".ruff_cache",
    ".svn",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "target",
    "venv",
}


@dataclass(frozen=True, slots=True)
class RepositoryIndexResult:
    """仓库索引结果。"""

    graph: GraphIR
    indexed_file_count: int


def index_repository(
    repository_path: Path,
    *,
    embedding_client: SummaryEmbeddingClient | RigelEmbedding | None = None,
    summary_client: SummaryTextClient | RigelLLM | None = None,
) -> RepositoryIndexResult:
    """扫描仓库源码，并通过真实 Java LSP 补全跨文件语义边。"""

    resolved_repository_path = repository_path.resolve()
    active_embedding_client = embedding_client or RigelEmbedding(
        EmbeddingConfig.from_repository(resolved_repository_path)
    )
    active_summary_client = summary_client or RigelLLM(
        LLMConfig.from_repository(resolved_repository_path, LLMConfigSection.SUMMARY)
    )
    repository_name = resolved_repository_path.name
    request = JavaParseRequest(repository_name=repository_name)
    graph = _base_graph(request)
    indexed_file_count = 0

    # 单文件解析会各自产生 Repository/Module 节点，合并时按 id 去重以保留解析器的自包含输出。
    for source_path in _iter_java_files(resolved_repository_path):
        relative_path = source_path.relative_to(resolved_repository_path).as_posix()
        file_graph = parse_java_file(source_path.read_bytes(), relative_path, request=request)
        _merge_graph(graph, file_graph)
        indexed_file_count += 1

    if indexed_file_count > 0:
        # 语义边需要跨文件视角，必须等所有文件的结构实体都进入同一个 GraphIR 后再补全。
        enrich_java_semantic_edges(
            graph,
            request=JavaSemanticEdgeRequest(repository_root_path=str(resolved_repository_path)),
        )
    attach_retrieval_summaries(
        graph,
        embedding_client=active_embedding_client,
        summary_client=active_summary_client,
    )

    return RepositoryIndexResult(graph=graph, indexed_file_count=indexed_file_count)


def _base_graph(request: JavaParseRequest) -> GraphIR:
    graph = GraphIR()
    repository = Repository(repo_id=f"repo:{request.repository_name}", name=request.repository_name)
    module = Module(
        module_id=f"module:{request.repository_name}:{DEFAULT_MODULE_NAME}",
        name=DEFAULT_MODULE_NAME,
        root_path=".",
        ecosystem=DEFAULT_MODULE_ECOSYSTEM,
        zone=DEFAULT_ZONE,
    )
    graph.add_node(repository)
    graph.add_node(module)
    graph.add_edge(
        GraphEdge.create(
            EdgeType.CONTAINS,
            repository.repo_id,
            module.module_id,
            kind="physical-membership",
        )
    )
    return graph


def _iter_java_files(repository_path: Path) -> list[Path]:
    # 排序让索引输出在不同文件系统遍历顺序下保持稳定，便于测试和演示复现。
    return sorted(
        path
        for path in repository_path.rglob("*.java")
        if path.is_file() and not _is_ignored_path(path.relative_to(repository_path))
    )


def _is_ignored_path(relative_path: Path) -> bool:
    return any(part in IGNORED_DIRECTORY_NAMES for part in relative_path.parts)


def _merge_graph(target: GraphIR, source: GraphIR) -> None:
    existing_node_ids = {node.id for node in target.nodes}
    existing_edge_ids = {edge.id for edge in target.edges}

    # GraphIR 当前是列表模型，这里显式维护 id 集合，避免 O(n) 反复查找放大仓库级合并成本。
    for node in source.nodes:
        if node.id in existing_node_ids:
            continue
        target.nodes.append(node)
        existing_node_ids.add(node.id)

    for edge in source.edges:
        if edge.id in existing_edge_ids:
            continue
        target.edges.append(edge)
        existing_edge_ids.add(edge.id)
