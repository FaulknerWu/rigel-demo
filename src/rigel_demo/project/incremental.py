"""仓库增量索引的文件变更与子图选择。"""

from __future__ import annotations

from dataclasses import dataclass

from rigel_demo.graph.ir import EdgeType, GraphEdge, GraphIR, GraphNode, NodeType
from rigel_demo.project.java_targets import JavaFileIndexTarget, content_hash

SEMANTIC_EDGE_TYPES = {EdgeType.DEPENDS_ON, EdgeType.SPECIALIZES, EdgeType.ALIASES}


@dataclass(frozen=True, slots=True)
class JavaFileChangeSet:
    """增量索引中的 Java 文件变更分类。"""

    added_files: list[str]
    modified_files: list[str]
    deleted_files: list[str]
    skipped_files: list[str]

    @property
    def changed_existing_files(self) -> list[str]:
        """需要重新解析和写入的新增或修改文件。"""

        return self.added_files + self.modified_files

    @property
    def changed_existing_file_paths(self) -> set[str]:
        """需要交给语义边补全器处理的新增或修改文件路径。"""

        return set(self.changed_existing_files)

    @property
    def indexed_file_count(self) -> int:
        """本次需要重新写入图谱的文件数量。"""

        return len(self.changed_existing_files)


def java_file_changes(
    targets: list[JavaFileIndexTarget],
    previous_file_hashes: dict[str, str],
) -> JavaFileChangeSet:
    current_file_hashes = _current_file_hashes(targets)
    current_paths = set(current_file_hashes)
    previous_paths = set(previous_file_hashes)
    stable_paths = current_paths & previous_paths

    return JavaFileChangeSet(
        added_files=sorted(current_paths - previous_paths),
        modified_files=sorted(
            path
            for path in stable_paths
            if current_file_hashes[path] != previous_file_hashes[path]
        ),
        deleted_files=sorted(previous_paths - current_paths),
        skipped_files=sorted(
            path
            for path in stable_paths
            if current_file_hashes[path] == previous_file_hashes[path]
        ),
    )


def incremental_node_ids(graph: GraphIR, changed_file_paths: set[str]) -> set[str]:
    nodes_by_id = {node.id: node for node in graph.nodes}
    file_ids = {
        node.id
        for node in graph.nodes
        if node.type == NodeType.FILE and node.properties["relative_path"] in changed_file_paths
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
        # 写入文件子图时保留 Repository/Module/File 祖先，保证数据库中可直接 MERGE 层级边。
        selected_node_ids.update(_ancestor_node_ids(file_id, parent_by_child, nodes_by_id))
        selected_node_ids.update(_descendant_node_ids(file_id, children_by_parent))

    for edge in graph.edges:
        if edge.type == EdgeType.HAS_ANCHOR and edge.source_id in selected_node_ids:
            # Anchor 不在 CONTAINS 树内，需要沿 HAS_ANCHOR 单独纳入增量写入集合。
            selected_node_ids.add(edge.target_id)
    return selected_node_ids


def summary_node_ids_for_targets(graph: GraphIR, target_node_ids: set[str]) -> set[str]:
    return {
        edge.source_id
        for edge in graph.edges
        if edge.type == EdgeType.DESCRIBES and edge.target_id in target_node_ids
    }


def select_incremental_graph(graph: GraphIR, selected_node_ids: set[str]) -> GraphIR:
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


def _current_file_hashes(targets: list[JavaFileIndexTarget]) -> dict[str, str]:
    return {
        target.relative_path: content_hash(target.source_path.read_bytes())
        for target in targets
    }


def _ancestor_node_ids(
    node_id: str,
    parent_by_child: dict[str, str],
    nodes_by_id: dict[str, GraphNode],
) -> set[str]:
    ancestors: set[str] = set()
    current_node_id: str | None = node_id
    while current_node_id is not None and current_node_id not in ancestors:
        node = nodes_by_id[current_node_id]
        if node.type in {NodeType.REPOSITORY, NodeType.MODULE, NodeType.FILE}:
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


def _should_select_incremental_edge(
    edge: GraphEdge,
    selected_node_ids: set[str],
    incremental_owner_node_ids: set[str],
) -> bool:
    if edge.source_id in selected_node_ids and edge.target_id in selected_node_ids:
        return True
    # 语义边可能连向未变化文件；保留这类一跳关系，避免增量后变更实体失去跨文件上下文。
    return (
        edge.type in SEMANTIC_EDGE_TYPES
        and (edge.source_id in incremental_owner_node_ids or edge.target_id in incremental_owner_node_ids)
    )
