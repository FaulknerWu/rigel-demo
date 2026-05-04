"""仓库级代码图谱索引流程。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rigel_demo.entities import Repository
from rigel_demo.graph.ir import GraphEdge, GraphIR, NodeType
from rigel_demo.embedding import EmbeddingConfig, RigelEmbedding
from rigel_demo.project.summaries import (
    SummaryEmbeddingClient,
    SummaryTextClient,
    attach_retrieval_summaries,
)
from rigel_demo.java import JavaParseRequest, JavaSemanticEdgeRequest, enrich_java_semantic_edges, parse_java_file
from rigel_demo.project.incremental import (
    incremental_node_ids,
    java_file_changes,
    select_incremental_graph,
    summary_node_ids_for_targets,
)
from rigel_demo.project.java_targets import (
    JavaFileIndexTarget,
    iter_java_targets,
)
from rigel_demo.llm import LLMConfig, LLMConfigSection, RigelLLM


@dataclass(frozen=True, slots=True)
class JavaStructureGraphResult:
    """Java 文件结构解析后的仓库级图谱。"""

    graph: GraphIR
    indexed_file_count: int


@dataclass(frozen=True, slots=True)
class RepositoryIndexResult:
    """仓库索引结果。"""

    graph: GraphIR
    indexed_file_count: int


@dataclass(frozen=True, slots=True)
class RepositoryIncrementalIndexResult:
    """仓库增量索引结果。"""

    graph: GraphIR
    added_files: list[str]
    modified_files: list[str]
    deleted_files: list[str]
    skipped_files: list[str]
    indexed_file_count: int

    @property
    def changed_file_count(self) -> int:
        """发生新增、修改或删除的文件数量。"""

        return len(self.added_files) + len(self.modified_files) + len(self.deleted_files)


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
    structure_result = _build_java_structure_graph(
        repository_name=resolved_repository_path.name,
        targets=iter_java_targets(resolved_repository_path),
    )

    if structure_result.indexed_file_count > 0:
        # 语义边需要跨文件视角，必须等所有文件的结构实体都进入同一个 GraphIR 后再补全。
        enrich_java_semantic_edges(
            structure_result.graph,
            request=JavaSemanticEdgeRequest(repository_root_path=str(resolved_repository_path)),
        )
    attach_retrieval_summaries(
        structure_result.graph,
        embedding_client=active_embedding_client,
        summary_client=active_summary_client,
    )

    return RepositoryIndexResult(
        graph=structure_result.graph,
        indexed_file_count=structure_result.indexed_file_count,
    )


def index_repository_incremental(
    repository_path: Path,
    *,
    previous_file_hashes: dict[str, str],
    embedding_client: SummaryEmbeddingClient | RigelEmbedding | None = None,
    summary_client: SummaryTextClient | RigelLLM | None = None,
) -> RepositoryIncrementalIndexResult:
    """构建新增和修改 Java 文件对应的可写入增量图谱。"""

    resolved_repository_path = repository_path.resolve()
    targets = iter_java_targets(resolved_repository_path)
    file_changes = java_file_changes(targets, previous_file_hashes)

    if not file_changes.changed_existing_files:
        return RepositoryIncrementalIndexResult(
            graph=GraphIR(),
            added_files=file_changes.added_files,
            modified_files=file_changes.modified_files,
            deleted_files=file_changes.deleted_files,
            skipped_files=file_changes.skipped_files,
            indexed_file_count=0,
        )

    active_embedding_client = embedding_client or RigelEmbedding(
        EmbeddingConfig.from_repository(resolved_repository_path)
    )
    active_summary_client = summary_client or RigelLLM(
        LLMConfig.from_repository(resolved_repository_path, LLMConfigSection.SUMMARY)
    )

    # 语义边依赖全仓实体索引；即使最终只写入变更文件，也需要用完整结构图解析跨文件目标。
    full_graph = _build_java_structure_graph(
        repository_name=resolved_repository_path.name,
        targets=targets,
    ).graph
    changed_file_paths = file_changes.changed_existing_file_paths
    enrich_java_semantic_edges(
        full_graph,
        request=JavaSemanticEdgeRequest(repository_root_path=str(resolved_repository_path)),
        source_file_paths=changed_file_paths,
        target_file_paths=changed_file_paths,
    )

    selected_node_ids = incremental_node_ids(full_graph, changed_file_paths)
    # Summary 只为本次要写入的结构节点生成，避免未变化文件重复调用模型。
    summary_target_node_ids = {
        node.id
        for node in full_graph.nodes
        if node.id in selected_node_ids and node.type in {NodeType.MODULE, NodeType.FILE, NodeType.ENTITY}
    }
    attach_retrieval_summaries(
        full_graph,
        embedding_client=active_embedding_client,
        summary_client=active_summary_client,
        target_node_ids=summary_target_node_ids,
    )
    selected_node_ids.update(summary_node_ids_for_targets(full_graph, summary_target_node_ids))

    return RepositoryIncrementalIndexResult(
        graph=select_incremental_graph(full_graph, selected_node_ids),
        added_files=file_changes.added_files,
        modified_files=file_changes.modified_files,
        deleted_files=file_changes.deleted_files,
        skipped_files=file_changes.skipped_files,
        indexed_file_count=file_changes.indexed_file_count,
    )


def _base_graph(repository_name: str) -> GraphIR:
    graph = GraphIR()
    repository = Repository(repo_id=f"repo:{repository_name}", name=repository_name)
    graph.add_node(repository)
    return graph


def _build_java_structure_graph(
    *,
    repository_name: str,
    targets: list[JavaFileIndexTarget],
) -> JavaStructureGraphResult:
    graph = _base_graph(repository_name)

    # 单文件解析会各自产生 Repository/Module 节点，合并时按 id 去重以保留解析器的自包含输出。
    for target in targets:
        request = _java_parse_request(repository_name, target)
        file_graph = parse_java_file(target.source_path.read_bytes(), target.relative_path, request=request)
        _merge_graph(graph, file_graph)

    return JavaStructureGraphResult(graph=graph, indexed_file_count=len(targets))


def _java_parse_request(repository_name: str, target: JavaFileIndexTarget) -> JavaParseRequest:
    return JavaParseRequest(
        repository_name=repository_name,
        module_name=target.module_name,
        module_root_path=target.module_root_path,
        module_ecosystem=target.module_ecosystem,
        zone=target.module_zone,
        file_zone=target.file_zone,
    )


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
