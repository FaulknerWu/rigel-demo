"""仓库级代码图谱索引流程。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rigel_demo.core.graph_ir import EdgeType, GraphEdge, GraphIR, GraphNode, NodeType, Repository
from rigel_demo.embedding import EmbeddingConfig, RigelEmbedding
from rigel_demo.indexing.retrieval_summaries import (
    SummaryEmbeddingClient,
    SummaryTextClient,
    attach_retrieval_summaries,
)
from rigel_demo.java import JavaParseRequest, JavaSemanticEdgeRequest, enrich_java_semantic_edges, parse_java_file
from rigel_demo.java.requests import (
    DEFAULT_MODULE_ECOSYSTEM,
    DEFAULT_MODULE_NAME,
    DEFAULT_ZONE,
    GENERATED_ZONE,
)
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
    "out",
    "target",
    "venv",
}

JAVA_SOURCE_ROOT_PATTERNS: tuple[tuple[str, ...], ...] = (
    ("src", "main", "java"),
    ("src", "test", "java"),
    ("src", "generated", "java"),
    ("generated", "src", "main", "java"),
    ("generated-sources",),
)
MAVEN_MARKER_FILE_NAME = "pom.xml"
GRADLE_MARKER_FILE_NAMES = {"build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts"}
TEST_PATH_PARTS = {"test", "tests", "it", "integrationtest", "integration-test"}
TOOLING_PATH_PARTS = {"tool", "tools", "tooling", "script", "scripts", "buildsrc", "build-logic"}
VENDOR_PATH_PARTS = {"vendor", "third_party", "third-party", "external"}
GENERATED_PATH_PARTS = {"generated", "generated-sources", "build", "out", "target"}
GENERATED_SOURCE_MARKER_PARTS = {"generated", "generated-sources"}
HASH_PREFIX = "sha256:"
SEMANTIC_EDGE_TYPES = {EdgeType.DEPENDS_ON, EdgeType.SPECIALIZES, EdgeType.ALIASES}


@dataclass(frozen=True, slots=True)
class JavaFileIndexTarget:
    """单个 Java 文件在仓库内的模块和分区归属。"""

    source_path: Path
    relative_path: str
    module_name: str
    module_root_path: str
    module_ecosystem: str
    module_zone: str
    file_zone: str


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
    repository_name = resolved_repository_path.name
    graph = _base_graph(repository_name)
    indexed_file_count = 0

    # 单文件解析会各自产生 Repository/Module 节点，合并时按 id 去重以保留解析器的自包含输出。
    for target in _iter_java_targets(resolved_repository_path):
        request = JavaParseRequest(
            repository_name=repository_name,
            module_name=target.module_name,
            module_root_path=target.module_root_path,
            module_ecosystem=target.module_ecosystem,
            zone=target.module_zone,
            file_zone=target.file_zone,
        )
        file_graph = parse_java_file(target.source_path.read_bytes(), target.relative_path, request=request)
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


def index_repository_incremental(
    repository_path: Path,
    *,
    previous_file_hashes: dict[str, str],
    embedding_client: SummaryEmbeddingClient | RigelEmbedding | None = None,
    summary_client: SummaryTextClient | RigelLLM | None = None,
) -> RepositoryIncrementalIndexResult:
    """构建新增和修改 Java 文件对应的可写入增量图谱。"""

    resolved_repository_path = repository_path.resolve()
    repository_name = resolved_repository_path.name
    targets = _iter_java_targets(resolved_repository_path)
    current_file_hashes = _current_file_hashes(targets)

    current_paths = set(current_file_hashes)
    previous_paths = set(previous_file_hashes)
    added_files = sorted(current_paths - previous_paths)
    modified_files = sorted(
        path
        for path in current_paths & previous_paths
        if current_file_hashes[path] != previous_file_hashes[path]
    )
    deleted_files = sorted(previous_paths - current_paths)
    skipped_files = sorted(
        path
        for path in current_paths & previous_paths
        if current_file_hashes[path] == previous_file_hashes[path]
    )
    changed_existing_files = added_files + modified_files

    if not changed_existing_files:
        return RepositoryIncrementalIndexResult(
            graph=GraphIR(),
            added_files=added_files,
            modified_files=modified_files,
            deleted_files=deleted_files,
            skipped_files=skipped_files,
            indexed_file_count=0,
        )

    active_embedding_client = embedding_client or RigelEmbedding(
        EmbeddingConfig.from_repository(resolved_repository_path)
    )
    active_summary_client = summary_client or RigelLLM(
        LLMConfig.from_repository(resolved_repository_path, LLMConfigSection.SUMMARY)
    )

    full_graph = _base_graph(repository_name)
    for target in targets:
        request = JavaParseRequest(
            repository_name=repository_name,
            module_name=target.module_name,
            module_root_path=target.module_root_path,
            module_ecosystem=target.module_ecosystem,
            zone=target.module_zone,
            file_zone=target.file_zone,
        )
        file_graph = parse_java_file(target.source_path.read_bytes(), target.relative_path, request=request)
        _merge_graph(full_graph, file_graph)

    changed_file_paths = set(changed_existing_files)
    enrich_java_semantic_edges(
        full_graph,
        request=JavaSemanticEdgeRequest(repository_root_path=str(resolved_repository_path)),
        source_file_paths=changed_file_paths,
        target_file_paths=changed_file_paths,
    )

    selected_node_ids = _incremental_node_ids(full_graph, changed_file_paths)
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
    selected_node_ids.update(_summary_node_ids_for_targets(full_graph, summary_target_node_ids))

    return RepositoryIncrementalIndexResult(
        graph=_select_incremental_graph(full_graph, selected_node_ids),
        added_files=added_files,
        modified_files=modified_files,
        deleted_files=deleted_files,
        skipped_files=skipped_files,
        indexed_file_count=len(changed_existing_files),
    )


def _base_graph(repository_name: str) -> GraphIR:
    graph = GraphIR()
    repository = Repository(repo_id=f"repo:{repository_name}", name=repository_name)
    graph.add_node(repository)
    return graph


def _iter_java_targets(repository_path: Path) -> list[JavaFileIndexTarget]:
    # 排序让索引输出在不同文件系统遍历顺序下保持稳定，便于测试和演示复现。
    return sorted(
        (
            _java_file_target(repository_path, path)
            for path in repository_path.rglob("*.java")
            if path.is_file() and not _is_ignored_path(path.relative_to(repository_path))
        ),
        key=lambda target: target.relative_path,
    )


def _is_ignored_path(relative_path: Path) -> bool:
    parts = {part.lower() for part in relative_path.parts}
    ignored_parts = parts & IGNORED_DIRECTORY_NAMES
    if not ignored_parts:
        return False
    if ignored_parts <= {"build", "out", "target"} and parts & GENERATED_SOURCE_MARKER_PARTS:
        return False
    return True


def _java_file_target(repository_path: Path, source_path: Path) -> JavaFileIndexTarget:
    relative_path = source_path.relative_to(repository_path)
    module_root = _detect_module_root(repository_path, relative_path)
    source_root = _detect_java_source_root(relative_path.relative_to(module_root))
    module_relative_path = relative_path.relative_to(module_root)
    file_zone = _detect_file_zone(module_relative_path, source_root=source_root)
    module_name = _module_name(module_root)
    module_zone = _module_zone(file_zone)
    return JavaFileIndexTarget(
        source_path=source_path,
        relative_path=relative_path.as_posix(),
        module_name=module_name,
        module_root_path=module_root.as_posix(),
        module_ecosystem=_detect_module_ecosystem(repository_path, module_root),
        module_zone=module_zone,
        file_zone=file_zone,
    )


def _detect_module_root(repository_path: Path, relative_path: Path) -> Path:
    parent_parts = relative_path.parts[:-1]
    for part_count in range(len(parent_parts), -1, -1):
        candidate = Path(*parent_parts[:part_count]) if part_count else Path(".")
        absolute_candidate = repository_path / candidate
        if _has_module_marker(absolute_candidate):
            return candidate
    return Path(".")


def _has_module_marker(path: Path) -> bool:
    return (path / MAVEN_MARKER_FILE_NAME).exists() or any((path / marker).exists() for marker in GRADLE_MARKER_FILE_NAMES)


def _detect_java_source_root(module_relative_path: Path) -> Path:
    parts = module_relative_path.parts
    for pattern in JAVA_SOURCE_ROOT_PATTERNS:
        pattern_length = len(pattern)
        if len(parts) >= pattern_length and tuple(part.lower() for part in parts[:pattern_length]) == pattern:
            return Path(*parts[:pattern_length])
    return Path(".")


def _detect_file_zone(module_relative_path: Path, *, source_root: Path) -> str:
    parts = {part.lower() for part in module_relative_path.parts}
    source_root_parts = {part.lower() for part in source_root.parts}
    if parts & GENERATED_PATH_PARTS or source_root_parts & GENERATED_PATH_PARTS:
        return GENERATED_ZONE
    if parts & VENDOR_PATH_PARTS:
        return "vendor"
    if parts & TOOLING_PATH_PARTS:
        return "tooling"
    if parts & TEST_PATH_PARTS:
        return "test"
    return DEFAULT_ZONE


def _module_zone(file_zone: str) -> Literal["prod", "test", "tooling", "vendor", "generated"]:
    if file_zone == "vendor":
        return "vendor"
    if file_zone == "tooling":
        return "tooling"
    return DEFAULT_ZONE


def _module_name(module_root: Path) -> str:
    if module_root == Path("."):
        return DEFAULT_MODULE_NAME
    return module_root.as_posix().replace("/", ":")


def _detect_module_ecosystem(repository_path: Path, module_root: Path) -> str:
    absolute_module_root = repository_path / module_root
    if (absolute_module_root / MAVEN_MARKER_FILE_NAME).exists():
        return DEFAULT_MODULE_ECOSYSTEM
    if any((absolute_module_root / marker).exists() for marker in GRADLE_MARKER_FILE_NAMES):
        return "gradle"
    return DEFAULT_MODULE_ECOSYSTEM


def _current_file_hashes(targets: list[JavaFileIndexTarget]) -> dict[str, str]:
    return {
        target.relative_path: _content_hash(target.source_path.read_bytes())
        for target in targets
    }


def _content_hash(content: bytes) -> str:
    return f"{HASH_PREFIX}{hashlib.sha256(content).hexdigest()}"


def _incremental_node_ids(graph: GraphIR, changed_file_paths: set[str]) -> set[str]:
    nodes_by_id = {node.id: node for node in graph.nodes}
    file_ids = {
        node.id
        for node in graph.nodes
        if node.type == NodeType.FILE and node.properties.get("relative_path") in changed_file_paths
    }
    children_by_parent: dict[str, list[str]] = {}
    parent_by_child: dict[str, str] = {}
    for edge in graph.edges:
        if edge.type != EdgeType.CONTAINS:
            continue
        children_by_parent.setdefault(edge.source_id, []).append(edge.target_id)
        parent_by_child[edge.target_id] = edge.source_id

    selected_node_ids: set[str] = set()
    for file_id in file_ids:
        selected_node_ids.update(_ancestor_node_ids(file_id, parent_by_child, nodes_by_id))
        selected_node_ids.update(_descendant_node_ids(file_id, children_by_parent))

    for edge in graph.edges:
        if edge.type == EdgeType.HAS_ANCHOR and edge.source_id in selected_node_ids:
            selected_node_ids.add(edge.target_id)
    return selected_node_ids


def _ancestor_node_ids(
    node_id: str,
    parent_by_child: dict[str, str],
    nodes_by_id: dict[str, GraphNode],
) -> set[str]:
    ancestors: set[str] = set()
    current_node_id: str | None = node_id
    while current_node_id is not None and current_node_id not in ancestors:
        node = nodes_by_id.get(current_node_id)
        if node is not None and node.type in {NodeType.REPOSITORY, NodeType.MODULE, NodeType.FILE}:
            ancestors.add(current_node_id)
        current_node_id = parent_by_child.get(current_node_id)
    return ancestors


def _descendant_node_ids(node_id: str, children_by_parent: dict[str, list[str]]) -> set[str]:
    descendants: set[str] = set()
    stack = [node_id]
    while stack:
        current_node_id = stack.pop()
        if current_node_id in descendants:
            continue
        descendants.add(current_node_id)
        stack.extend(children_by_parent.get(current_node_id, []))
    return descendants


def _summary_node_ids_for_targets(graph: GraphIR, target_node_ids: set[str]) -> set[str]:
    return {
        edge.source_id
        for edge in graph.edges
        if edge.type == EdgeType.DESCRIBES and edge.target_id in target_node_ids
    }


def _select_incremental_graph(graph: GraphIR, selected_node_ids: set[str]) -> GraphIR:
    selected_nodes = [
        node
        for node in graph.nodes
        if node.id in selected_node_ids
    ]
    incremental_owner_node_ids = {
        node.id
        for node in selected_nodes
        if node.type in {NodeType.FILE, NodeType.ENTITY}
    }
    selected_edges = [
        edge
        for edge in graph.edges
        if _should_select_incremental_edge(edge, selected_node_ids, incremental_owner_node_ids)
    ]
    return GraphIR(nodes=selected_nodes, edges=selected_edges)


def _should_select_incremental_edge(
    edge: GraphEdge,
    selected_node_ids: set[str],
    incremental_owner_node_ids: set[str],
) -> bool:
    if edge.source_id in selected_node_ids and edge.target_id in selected_node_ids:
        return True
    return (
        edge.type in SEMANTIC_EDGE_TYPES
        and (edge.source_id in incremental_owner_node_ids or edge.target_id in incremental_owner_node_ids)
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
