"""基于 Java LSP 与 Tree-sitter 的跨文件语义边补全。"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import ContextManager, Protocol

from multilspy import SyncLanguageServer
from multilspy.multilspy_config import MultilspyConfig
from multilspy.multilspy_logger import MultilspyLogger
from tree_sitter import Node, Parser

from rigel_demo.core.graph_ir import EdgeType, GraphEdge, GraphIR, JsonObject
from rigel_demo.java.graph_index import EntityView, GraphIndex
from rigel_demo.java.language import (
    DECLARATION_NODE_KINDS,
    INHERITANCE_CONTAINER_KINDS,
    JAVA_LANGUAGE,
    TREE_SITTER_PROVENANCE,
    TYPE_REFERENCE_NODE_KINDS,
)
from rigel_demo.java.requests import JavaSemanticEdgeRequest
from rigel_demo.java.source_utils import (
    find_import_node,
    has_ancestor_until_declaration,
    has_parent,
    last_named_child,
    normalize_path,
    read_imports,
)

LSP_PROVENANCE = "lsp"
LSP_CONFIDENCE = 0.95
TREE_SITTER_CONFIDENCE = 0.55

ALIAS_GENERATED_MIRROR_KIND = "generated-mirror"
ALIAS_DUPLICATE_ENTITY_KIND = "duplicate-entity"
DEPENDENCY_IMPORT_KIND = "imports"
DEPENDENCY_CALL_KIND = "calls"
DEPENDENCY_REFERENCE_KIND = "references"
DEPENDENCY_TYPE_USE_KIND = "type-use"
SPECIALIZES_EXTENDS_KIND = "extends"
SPECIALIZES_IMPLEMENTS_KIND = "implements"
SPECIALIZES_OVERRIDE_KIND = "override"


class JavaLspClient(Protocol):
    """语义边补全依赖的最小 LSP 客户端接口。"""

    def request_definition(self, file_path: str, line: int, column: int) -> list[JsonObject]: ...

    def request_references(self, file_path: str, line: int, column: int) -> list[JsonObject]: ...


@dataclass(frozen=True, slots=True)
class _SemanticCandidate:
    source_entity_id: str
    edge_type: EdgeType
    kind: str
    file_path: str
    line: int
    column: int


def enrich_java_semantic_edges(
    graph: GraphIR,
    *,
    request: JavaSemanticEdgeRequest,
    lsp_client: JavaLspClient | None = None,
) -> GraphIR:
    """在已有结构图上补全 Java 跨文件语义边。

    当传入 lsp_client 时复用外部客户端，便于测试或上层统一管理 LSP 生命周期；否则
    使用 multilspy 启动 Java LSP。跨文件依赖只接受 LSP 定义跳转确认后的目标。
    """

    repository_root = Path(request.repository_root_path).resolve()
    graph_index = GraphIndex(graph)
    lsp_context = _started_lsp(repository_root, request.lsp_timeout_seconds) if lsp_client is None else nullcontext(lsp_client)

    with lsp_context as active_lsp:
        # 候选边先由 Tree-sitter 定位语法位置，再交给 LSP 解析真实目标，兼顾覆盖率和语义精度。
        for candidate in _collect_tree_sitter_candidates(repository_root, graph_index):
            target_entity = _resolve_lsp_target(active_lsp, repository_root, graph_index, candidate)
            if target_entity is not None and target_entity.node.id != candidate.source_entity_id:
                _add_semantic_edge(
                    graph,
                    candidate,
                    target_entity.node.id,
                    provenance=LSP_PROVENANCE,
                    confidence=LSP_CONFIDENCE,
                )

        # references 与 override 依赖完整实体索引，放在候选边补全后统一追加，避免重复扫描 AST。
        _add_lsp_reference_edges(graph, active_lsp, repository_root, graph_index)
        _add_override_edges(graph, graph_index)
        _add_alias_edges(graph, graph_index)

    return graph


def _started_lsp(repository_root: Path, timeout_seconds: int) -> ContextManager[JavaLspClient]:
    config = MultilspyConfig.from_dict({"code_language": "java"})
    language_server = SyncLanguageServer.create(config, MultilspyLogger(), str(repository_root), timeout=timeout_seconds)
    return language_server.start_server()


def _collect_tree_sitter_candidates(repository_root: Path, graph_index: GraphIndex) -> list[_SemanticCandidate]:
    candidates: list[_SemanticCandidate] = []
    parser = Parser()
    parser.language = JAVA_LANGUAGE

    for file_path in graph_index.java_file_paths:
        source_path = repository_root / file_path
        source_bytes = source_path.read_bytes()
        root_node = parser.parse(source_bytes).root_node
        file_entities = graph_index.entities_by_file_path.get(file_path, [])
        top_level_entities = [entity for entity in file_entities if _is_top_level_entity(entity)]

        # import 属于文件级语义，这里挂到顶层实体上，避免把依赖关系散落到无源码实体的文件节点。
        for imported_name in read_imports(root_node, source_bytes).values():
            for source_entity in top_level_entities:
                import_node = find_import_node(root_node, source_bytes, imported_name)
                if import_node is None:
                    continue
                candidates.append(
                    _candidate(
                        source_entity.node.id,
                        EdgeType.DEPENDS_ON,
                        DEPENDENCY_IMPORT_KIND,
                        file_path,
                        import_node,
                    )
                )

        _walk_candidates(
            root_node,
            graph_index,
            file_path,
            candidates,
        )
    return candidates


def _walk_candidates(
    node: Node,
    graph_index: GraphIndex,
    file_path: str,
    candidates: list[_SemanticCandidate],
) -> None:
    owner = graph_index.find_owner_entity(file_path, node.start_point.row + 1, node.start_point.column + 1)
    owner_id = owner.node.id if owner is not None else None

    if owner_id is not None and node.type == "method_invocation":
        name_node = node.child_by_field_name("name") or last_named_child(node, "identifier")
        if name_node is not None:
            candidates.append(
                _candidate(
                    owner_id,
                    EdgeType.DEPENDS_ON,
                    DEPENDENCY_CALL_KIND,
                    file_path,
                    name_node,
                )
            )

    # 继承/实现会在专门分支生成 SPECIALIZES 边，这里排除这些容器以免同一类型同时产生依赖边。
    if owner_id is not None and node.type in TYPE_REFERENCE_NODE_KINDS and not has_ancestor_until_declaration(node, INHERITANCE_CONTAINER_KINDS, DECLARATION_NODE_KINDS):
        candidates.append(
            _candidate(
                owner_id,
                EdgeType.DEPENDS_ON,
                DEPENDENCY_TYPE_USE_KIND,
                file_path,
                node,
            )
        )

    if owner_id is not None and has_parent(node, "superclass") and node.type in TYPE_REFERENCE_NODE_KINDS:
        candidates.append(
            _candidate(
                owner_id,
                EdgeType.SPECIALIZES,
                SPECIALIZES_EXTENDS_KIND,
                file_path,
                node,
            )
        )

    if owner_id is not None and has_ancestor_until_declaration(node, {"super_interfaces", "extends_interfaces"}, DECLARATION_NODE_KINDS) and node.type in TYPE_REFERENCE_NODE_KINDS:
        candidates.append(
            _candidate(
                owner_id,
                EdgeType.SPECIALIZES,
                SPECIALIZES_IMPLEMENTS_KIND,
                file_path,
                node,
            )
        )

    for child in node.named_children:
        _walk_candidates(child, graph_index, file_path, candidates)


def _add_lsp_reference_edges(
    graph: GraphIR,
    lsp_client: JavaLspClient,
    repository_root: Path,
    graph_index: GraphIndex,
) -> None:
    for target_entity in graph_index.entities:
        anchor = target_entity.name_anchor or target_entity.definition_anchor
        if anchor is None:
            continue
        # GraphIR 对外使用 1 基坐标；LSP 协议使用 0 基坐标，请求前必须还原。
        line = int(anchor.properties["start_line"]) - 1
        column = _lsp_utf16_column(
            repository_root,
            target_entity.file_path,
            zero_based_line=line,
            zero_based_byte_column=int(anchor.properties["start_col"]) - 1,
        )
        for location in _lsp_locations(lsp_client.request_references, target_entity.file_path, line, column):
            source_entity = graph_index.find_location_owner(location)
            if source_entity is None or source_entity.node.id == target_entity.node.id:
                continue
            _add_graph_edge_once(
                graph,
                GraphEdge.create(
                    EdgeType.DEPENDS_ON,
                    source_entity.node.id,
                    target_entity.node.id,
                    kind=DEPENDENCY_REFERENCE_KIND,
                    provenance=LSP_PROVENANCE,
                    confidence=LSP_CONFIDENCE,
                ),
            )


def _add_override_edges(graph: GraphIR, graph_index: GraphIndex) -> None:
    specializes_edges = [edge for edge in graph.edges if edge.type == EdgeType.SPECIALIZES]
    # 当前 demo 只推导直接父类型上的同名方法覆盖关系，避免在缺少完整类型系统时过度猜测。
    parent_by_child = {
        edge.source_id: edge.target_id
        for edge in specializes_edges
        if edge.properties.get("kind") in {SPECIALIZES_EXTENDS_KIND, SPECIALIZES_IMPLEMENTS_KIND}
    }
    method_entities = [entity for entity in graph_index.entities if entity.node.properties.get("kind_norm") == "method"]
    methods_by_parent_and_name: dict[tuple[str, str], list[EntityView]] = {}
    for method_entity in method_entities:
        parent_name, method_name = _split_method_owner_and_name(str(method_entity.node.properties["qualified_name"]))
        parent_entity = graph_index.find_unique_entity_by_qualified_name(parent_name)
        if parent_entity is None:
            continue
        methods_by_parent_and_name.setdefault((parent_entity.node.id, method_name), []).append(method_entity)

    for method_entity in method_entities:
        parent_name, method_name = _split_method_owner_and_name(str(method_entity.node.properties["qualified_name"]))
        parent_entity = graph_index.find_unique_entity_by_qualified_name(parent_name)
        if parent_entity is None or parent_entity.node.id not in parent_by_child:
            continue
        overridden_methods = methods_by_parent_and_name.get((parent_by_child[parent_entity.node.id], method_name), [])
        for overridden_method in overridden_methods:
            _add_graph_edge_once(
                graph,
                GraphEdge.create(
                    EdgeType.SPECIALIZES,
                    method_entity.node.id,
                    overridden_method.node.id,
                    kind=SPECIALIZES_OVERRIDE_KIND,
                    provenance=TREE_SITTER_PROVENANCE,
                    confidence=TREE_SITTER_CONFIDENCE,
                ),
            )


def _add_alias_edges(graph: GraphIR, graph_index: GraphIndex) -> None:
    """为同一语义身份的多份实体生成 ALIASES 边。"""

    entities_by_key: dict[str, list[EntityView]] = {}
    for entity in graph_index.entities:
        entity_key = str(entity.node.properties.get("entity_key", ""))
        if entity_key:
            entities_by_key.setdefault(entity_key, []).append(entity)

    for aliased_entities in entities_by_key.values():
        if len(aliased_entities) < 2:
            continue
        canonical_entity = _canonical_alias_entity(aliased_entities)
        for aliased_entity in aliased_entities:
            if aliased_entity.node.id == canonical_entity.node.id:
                continue
            kind = (
                ALIAS_GENERATED_MIRROR_KIND
                if aliased_entity.node.properties.get("origin") == "generated"
                or canonical_entity.node.properties.get("origin") == "generated"
                else ALIAS_DUPLICATE_ENTITY_KIND
            )
            _add_graph_edge_once(
                graph,
                GraphEdge.create(
                    EdgeType.ALIASES,
                    aliased_entity.node.id,
                    canonical_entity.node.id,
                    kind=kind,
                    provenance=TREE_SITTER_PROVENANCE,
                    confidence=1.0,
                ),
            )


def _canonical_alias_entity(entities: list[EntityView]) -> EntityView:
    def sort_key(entity: EntityView) -> tuple[int, str]:
        origin = str(entity.node.properties.get("origin", "internal"))
        origin_priority = 1 if origin == "generated" else 0
        return origin_priority, entity.file_path

    return sorted(entities, key=sort_key)[0]


def _resolve_lsp_target(
    lsp_client: JavaLspClient,
    repository_root: Path,
    graph_index: GraphIndex,
    candidate: _SemanticCandidate,
) -> EntityView | None:
    column = _lsp_utf16_column(
        repository_root,
        candidate.file_path,
        zero_based_line=candidate.line,
        zero_based_byte_column=candidate.column,
    )
    locations = _lsp_locations(lsp_client.request_definition, candidate.file_path, candidate.line, column)
    for location in locations:
        target_entity = graph_index.find_location_target(location)
        if target_entity is not None:
            return target_entity
    return None


def _lsp_locations(method: object, file_path: str, line: int, column: int) -> list[JsonObject]:
    result = method(file_path, line, column)  # type: ignore[misc]
    return [dict(location) for location in result]


def _lsp_utf16_column(
    repository_root: Path,
    file_path: str,
    *,
    zero_based_line: int,
    zero_based_byte_column: int,
) -> int:
    """把 Tree-sitter 的 UTF-8 字节列转换成 LSP 默认使用的 UTF-16 列。"""

    source_path = repository_root / normalize_path(file_path)
    source_lines = source_path.read_bytes().splitlines()
    if zero_based_line < 0 or zero_based_line >= len(source_lines):
        raise ValueError(f"LSP 坐标行号超出源码范围：{file_path}:{zero_based_line + 1}")

    line_prefix = source_lines[zero_based_line][:zero_based_byte_column]
    decoded_prefix = line_prefix.decode("utf-8")
    return len(decoded_prefix.encode("utf-16-le")) // 2


def _add_semantic_edge(
    graph: GraphIR,
    candidate: _SemanticCandidate,
    target_entity_id: str,
    *,
    provenance: str,
    confidence: float,
) -> None:
    _add_graph_edge_once(
        graph,
        GraphEdge.create(
            candidate.edge_type,
            candidate.source_entity_id,
            target_entity_id,
            kind=candidate.kind,
            provenance=provenance,
            confidence=confidence,
        ),
    )


def _add_graph_edge_once(graph: GraphIR, edge: GraphEdge) -> None:
    if any(existing_edge.id == edge.id for existing_edge in graph.edges):
        return
    graph.add_edge(edge)


def _candidate(
    source_entity_id: str,
    edge_type: EdgeType,
    kind: str,
    file_path: str,
    node: Node,
) -> _SemanticCandidate:
    return _SemanticCandidate(
        source_entity_id=source_entity_id,
        edge_type=edge_type,
        kind=kind,
        file_path=file_path,
        line=node.start_point.row,
        column=node.start_point.column,
    )


def _is_top_level_entity(entity: EntityView) -> bool:
    qualified_name = str(entity.node.properties["qualified_name"])
    return "#" not in qualified_name and qualified_name.count(".") <= 2


def _split_method_owner_and_name(qualified_name: str) -> tuple[str, str]:
    owner_name, _, method_signature = qualified_name.partition("#")
    method_name = method_signature.split("(", 1)[0]
    return owner_name, method_name
