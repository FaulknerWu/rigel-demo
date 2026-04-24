"""基于 Java LSP 与 Tree-sitter 的跨文件语义边补全。"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import ContextManager, Protocol

from multilspy import SyncLanguageServer
from multilspy.multilspy_config import MultilspyConfig
from multilspy.multilspy_logger import MultilspyLogger
from tree_sitter import Node, Parser

from rigel_demo.graph_ir import EdgeType, GraphEdge, GraphIR, GraphNode, JsonObject, NodeType
from rigel_demo.java_parser import JAVA_LANGUAGE, TREE_SITTER_PROVENANCE


LSP_PROVENANCE = "lsp"
LSP_CONFIDENCE = 0.95
TREE_SITTER_CONFIDENCE = 0.55

DEPENDENCY_IMPORT_KIND = "imports"
DEPENDENCY_CALL_KIND = "calls"
DEPENDENCY_REFERENCE_KIND = "references"
DEPENDENCY_TYPE_USE_KIND = "type-use"
SPECIALIZES_EXTENDS_KIND = "extends"
SPECIALIZES_IMPLEMENTS_KIND = "implements"
SPECIALIZES_OVERRIDE_KIND = "override"

TYPE_REFERENCE_NODE_KINDS = {
    "array_type",
    "boolean_type",
    "floating_point_type",
    "generic_type",
    "integral_type",
    "scoped_type_identifier",
    "type_identifier",
    "void_type",
}
INHERITANCE_CONTAINER_KINDS = {"superclass", "super_interfaces", "extends_interfaces"}
DECLARATION_NODE_KINDS = {
    "class_declaration",
    "constructor_declaration",
    "enum_declaration",
    "interface_declaration",
    "method_declaration",
    "record_declaration",
}


class JavaLspClient(Protocol):
    """语义边补全依赖的最小 LSP 客户端接口。"""

    def request_definition(self, file_path: str, line: int, column: int) -> list[JsonObject]: ...

    def request_references(self, file_path: str, line: int, column: int) -> list[JsonObject]: ...

    def request_hover(self, relative_file_path: str, line: int, column: int) -> JsonObject | None: ...


@dataclass(frozen=True, slots=True)
class JavaSemanticEdgeRequest:
    """Java 语义边补全请求。"""

    repository_root_path: str
    lsp_timeout_seconds: int = 30


@dataclass(frozen=True, slots=True)
class _EntityView:
    node: GraphNode
    file_path: str
    definition_anchor: GraphNode | None
    body_anchor: GraphNode | None


@dataclass(frozen=True, slots=True)
class _SemanticCandidate:
    source_entity_id: str
    edge_type: EdgeType
    kind: str
    file_path: str
    line: int
    column: int
    fallback_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _OwnerInterval:
    entity_id: str
    start_line: int
    start_col: int
    end_line: int
    end_col: int


def enrich_java_semantic_edges(
    graph: GraphIR,
    *,
    request: JavaSemanticEdgeRequest,
    lsp_client: JavaLspClient | None = None,
) -> GraphIR:
    """在已有结构图上补全 Java 跨文件语义边。

    当传入 lsp_client 时复用外部客户端，便于测试或上层统一管理 LSP 生命周期；否则
    使用 multilspy 启动 Java LSP。LSP 成功定位的边使用较高置信度，无法定位时再使用
    Tree-sitter 名称推断并标记较低置信度。
    """

    repository_root = Path(request.repository_root_path).resolve()
    graph_index = _GraphIndex(graph)
    lsp_context = _started_lsp(repository_root, request.lsp_timeout_seconds) if lsp_client is None else nullcontext(lsp_client)

    with lsp_context as active_lsp:
        for candidate in _collect_tree_sitter_candidates(repository_root, graph_index):
            target_entity = _resolve_lsp_target(active_lsp, graph_index, candidate)
            if target_entity is not None and target_entity.node.id != candidate.source_entity_id:
                _add_semantic_edge(
                    graph,
                    candidate,
                    target_entity.node.id,
                    provenance=LSP_PROVENANCE,
                    confidence=LSP_CONFIDENCE,
                )
                continue

            fallback_target = graph_index.find_unique_entity_by_names(candidate.fallback_names)
            if fallback_target is not None and fallback_target.node.id != candidate.source_entity_id:
                _add_semantic_edge(
                    graph,
                    candidate,
                    fallback_target.node.id,
                    provenance=TREE_SITTER_PROVENANCE,
                    confidence=TREE_SITTER_CONFIDENCE,
                )

        _add_lsp_reference_edges(graph, active_lsp, graph_index)
        _add_override_edges(graph, graph_index)

    return graph


def _started_lsp(repository_root: Path, timeout_seconds: int) -> ContextManager[JavaLspClient]:
    config = MultilspyConfig.from_dict({"code_language": "java"})
    language_server = SyncLanguageServer.create(config, MultilspyLogger(), str(repository_root), timeout=timeout_seconds)
    return language_server.start_server()


def _collect_tree_sitter_candidates(repository_root: Path, graph_index: "_GraphIndex") -> list[_SemanticCandidate]:
    candidates: list[_SemanticCandidate] = []
    parser = Parser()
    parser.language = JAVA_LANGUAGE

    for file_path in graph_index.java_file_paths:
        source_path = repository_root / file_path
        if not source_path.exists():
            continue
        source_bytes = source_path.read_bytes()
        root_node = parser.parse(source_bytes).root_node
        package_name = _read_package_name(root_node, source_bytes)
        imports = _read_imports(root_node, source_bytes)
        file_entities = graph_index.entities_by_file_path.get(file_path, [])
        top_level_entities = [entity for entity in file_entities if _is_top_level_entity(entity)]

        for imported_name in imports.values():
            for source_entity in top_level_entities:
                import_node = _find_import_node(root_node, source_bytes, imported_name)
                if import_node is None:
                    continue
                candidates.append(
                    _candidate(
                        source_entity.node.id,
                        EdgeType.DEPENDS_ON,
                        DEPENDENCY_IMPORT_KIND,
                        file_path,
                        import_node,
                        (imported_name,),
                    )
                )

        _walk_candidates(
            root_node,
            source_bytes,
            graph_index,
            file_path,
            package_name,
            imports,
            candidates,
        )
    return candidates


def _walk_candidates(
    node: Node,
    source_bytes: bytes,
    graph_index: "_GraphIndex",
    file_path: str,
    package_name: str,
    imports: dict[str, str],
    candidates: list[_SemanticCandidate],
) -> None:
    owner = graph_index.find_owner_entity(file_path, node.start_point.row + 1, node.start_point.column + 1)
    owner_id = owner.node.id if owner is not None else None

    if owner_id is not None and node.type == "method_invocation":
        name_node = node.child_by_field_name("name") or _last_named_child(node, "identifier")
        if name_node is not None:
            method_name = _node_text(name_node, source_bytes)
            candidates.append(
                _candidate(
                    owner_id,
                    EdgeType.DEPENDS_ON,
                    DEPENDENCY_CALL_KIND,
                    file_path,
                    name_node,
                    (method_name,),
                )
            )

    if owner_id is not None and node.type in TYPE_REFERENCE_NODE_KINDS and not _has_ancestor(node, INHERITANCE_CONTAINER_KINDS):
        type_name = _node_text(node, source_bytes)
        candidates.append(
            _candidate(
                owner_id,
                EdgeType.DEPENDS_ON,
                DEPENDENCY_TYPE_USE_KIND,
                file_path,
                node,
                _type_fallback_names(type_name, package_name, imports),
            )
        )

    if owner_id is not None and _has_parent(node, "superclass") and node.type in TYPE_REFERENCE_NODE_KINDS:
        type_name = _node_text(node, source_bytes)
        candidates.append(
            _candidate(
                owner_id,
                EdgeType.SPECIALIZES,
                SPECIALIZES_EXTENDS_KIND,
                file_path,
                node,
                _type_fallback_names(type_name, package_name, imports),
            )
        )

    if owner_id is not None and _has_ancestor(node, {"super_interfaces", "extends_interfaces"}) and node.type in TYPE_REFERENCE_NODE_KINDS:
        type_name = _node_text(node, source_bytes)
        candidates.append(
            _candidate(
                owner_id,
                EdgeType.SPECIALIZES,
                SPECIALIZES_IMPLEMENTS_KIND,
                file_path,
                node,
                _type_fallback_names(type_name, package_name, imports),
            )
        )

    for child in node.named_children:
        _walk_candidates(child, source_bytes, graph_index, file_path, package_name, imports, candidates)


def _add_lsp_reference_edges(graph: GraphIR, lsp_client: JavaLspClient, graph_index: "_GraphIndex") -> None:
    for target_entity in graph_index.entities:
        anchor = target_entity.definition_anchor
        if anchor is None:
            continue
        line = int(anchor.properties["start_line"]) - 1
        column = int(anchor.properties["start_col"]) - 1
        for location in _safe_lsp_locations(lsp_client.request_references, target_entity.file_path, line, column):
            source_entity = graph_index.find_location_owner(location)
            if source_entity is None or source_entity.node.id == target_entity.node.id:
                continue
            graph.add_edge(
                GraphEdge.create(
                    EdgeType.DEPENDS_ON,
                    source_entity.node.id,
                    target_entity.node.id,
                    kind=DEPENDENCY_REFERENCE_KIND,
                    provenance=LSP_PROVENANCE,
                    confidence=LSP_CONFIDENCE,
                )
            )


def _add_override_edges(graph: GraphIR, graph_index: "_GraphIndex") -> None:
    specializes_edges = [edge for edge in graph.edges if edge.type == EdgeType.SPECIALIZES]
    parent_by_child = {
        edge.source_id: edge.target_id
        for edge in specializes_edges
        if edge.properties.get("kind") in {SPECIALIZES_EXTENDS_KIND, SPECIALIZES_IMPLEMENTS_KIND}
    }
    method_entities = [entity for entity in graph_index.entities if entity.node.properties.get("kind_norm") == "method"]
    methods_by_parent_and_name: dict[tuple[str, str], list[_EntityView]] = {}
    for method_entity in method_entities:
        parent_name, method_name = _split_method_owner_and_name(str(method_entity.node.properties["qualified_name"]))
        parent_entity = graph_index.find_unique_entity_by_names((parent_name,))
        if parent_entity is None:
            continue
        methods_by_parent_and_name.setdefault((parent_entity.node.id, method_name), []).append(method_entity)

    for method_entity in method_entities:
        parent_name, method_name = _split_method_owner_and_name(str(method_entity.node.properties["qualified_name"]))
        parent_entity = graph_index.find_unique_entity_by_names((parent_name,))
        if parent_entity is None or parent_entity.node.id not in parent_by_child:
            continue
        overridden_methods = methods_by_parent_and_name.get((parent_by_child[parent_entity.node.id], method_name), [])
        for overridden_method in overridden_methods:
            graph.add_edge(
                GraphEdge.create(
                    EdgeType.SPECIALIZES,
                    method_entity.node.id,
                    overridden_method.node.id,
                    kind=SPECIALIZES_OVERRIDE_KIND,
                    provenance=TREE_SITTER_PROVENANCE,
                    confidence=TREE_SITTER_CONFIDENCE,
                )
            )


def _resolve_lsp_target(
    lsp_client: JavaLspClient,
    graph_index: "_GraphIndex",
    candidate: _SemanticCandidate,
) -> _EntityView | None:
    locations = _safe_lsp_locations(lsp_client.request_definition, candidate.file_path, candidate.line, candidate.column)
    for location in locations:
        target_entity = graph_index.find_location_target(location)
        if target_entity is not None:
            return target_entity
    return None


def _safe_lsp_locations(method: object, file_path: str, line: int, column: int) -> list[JsonObject]:
    try:
        result = method(file_path, line, column)  # type: ignore[misc]
    except Exception:
        return []
    return [dict(location) for location in result]


def _add_semantic_edge(
    graph: GraphIR,
    candidate: _SemanticCandidate,
    target_entity_id: str,
    *,
    provenance: str,
    confidence: float,
) -> None:
    graph.add_edge(
        GraphEdge.create(
            candidate.edge_type,
            candidate.source_entity_id,
            target_entity_id,
            kind=candidate.kind,
            provenance=provenance,
            confidence=confidence,
        )
    )


def _candidate(
    source_entity_id: str,
    edge_type: EdgeType,
    kind: str,
    file_path: str,
    node: Node,
    fallback_names: tuple[str, ...],
) -> _SemanticCandidate:
    return _SemanticCandidate(
        source_entity_id=source_entity_id,
        edge_type=edge_type,
        kind=kind,
        file_path=file_path,
        line=node.start_point.row,
        column=node.start_point.column,
        fallback_names=fallback_names,
    )


class _GraphIndex:
    def __init__(self, graph: GraphIR) -> None:
        self.nodes_by_id = {node.id: node for node in graph.nodes}
        self.file_path_by_file_id = {
            node.id: str(node.properties["relative_path"])
            for node in graph.nodes
            if node.type == NodeType.FILE and node.properties.get("language") == "java"
        }
        self.java_file_paths = list(self.file_path_by_file_id.values())
        self.anchor_by_id = {node.id: node for node in graph.nodes if node.type == NodeType.ANCHOR}
        self.anchors_by_owner_and_role = self._index_anchors(graph)
        self.entities = self._index_entities(graph)
        self.entities_by_file_path = self._group_entities_by_file_path()
        self.entities_by_qualified_name = self._group_entities_by_property("qualified_name")
        self.entities_by_display_name = self._group_entities_by_property("display_name")
        self.owner_intervals_by_file = self._index_owner_intervals()

    def find_unique_entity_by_names(self, names: tuple[str, ...]) -> _EntityView | None:
        for name in names:
            entities = self.entities_by_qualified_name.get(name) or self.entities_by_display_name.get(_simple_name(name), [])
            if len(entities) == 1:
                return entities[0]
        return None

    def find_location_target(self, location: JsonObject) -> _EntityView | None:
        file_path, line, column = _location_position(location)
        containing_entities: list[tuple[_EntityView, tuple[int, int]]] = []
        for entity in self.entities_by_file_path.get(file_path, []):
            anchor = entity.definition_anchor or entity.body_anchor
            if anchor is not None and _anchor_contains(anchor, line + 1, column + 1):
                interval = _OwnerInterval(
                    entity_id=entity.node.id,
                    start_line=int(anchor.properties["start_line"]),
                    start_col=int(anchor.properties["start_col"]),
                    end_line=int(anchor.properties["end_line"]),
                    end_col=int(anchor.properties["end_col"]),
                )
                containing_entities.append((entity, _span_size(interval)))
        if containing_entities:
            containing_entities.sort(key=lambda item: item[1])
            return containing_entities[0][0]
        return self.find_owner_entity(file_path, line + 1, column + 1)

    def find_location_owner(self, location: JsonObject) -> _EntityView | None:
        file_path, line, column = _location_position(location)
        return self.find_owner_entity(file_path, line + 1, column + 1)

    def find_owner_entity(self, file_path: str, line: int, column: int) -> _EntityView | None:
        intervals = self.owner_intervals_by_file.get(_normalize_path(file_path), [])
        for interval in intervals:
            if _span_contains(interval.start_line, interval.start_col, interval.end_line, interval.end_col, line, column):
                return self._entity_by_id(interval.entity_id)
        return None

    def _entity_by_id(self, entity_id: str) -> _EntityView | None:
        return next((entity for entity in self.entities if entity.node.id == entity_id), None)

    def _index_anchors(self, graph: GraphIR) -> dict[tuple[str, str], GraphNode]:
        anchors: dict[tuple[str, str], GraphNode] = {}
        for edge in graph.edges:
            if edge.type != EdgeType.HAS_ANCHOR:
                continue
            role = str(edge.properties.get("role", ""))
            anchor = self.anchor_by_id.get(edge.target_id)
            if anchor is not None:
                anchors[(edge.source_id, role)] = anchor
        return anchors

    def _index_entities(self, graph: GraphIR) -> list[_EntityView]:
        entities: list[_EntityView] = []
        parent_by_child = {
            edge.target_id: edge.source_id
            for edge in graph.edges
            if edge.type == EdgeType.CONTAINS and edge.target_id in self.nodes_by_id
        }
        for node in graph.nodes:
            if node.type != NodeType.ENTITY:
                continue
            file_id = _find_parent_file_id(node.id, parent_by_child, self.file_path_by_file_id)
            if file_id is None:
                continue
            entities.append(
                _EntityView(
                    node=node,
                    file_path=self.file_path_by_file_id[file_id],
                    definition_anchor=self.anchors_by_owner_and_role.get((node.id, "definition")),
                    body_anchor=self.anchors_by_owner_and_role.get((node.id, "body")),
                )
            )
        return entities

    def _group_entities_by_file_path(self) -> dict[str, list[_EntityView]]:
        grouped: dict[str, list[_EntityView]] = {}
        for entity in self.entities:
            grouped.setdefault(entity.file_path, []).append(entity)
        return grouped

    def _group_entities_by_property(self, property_name: str) -> dict[str, list[_EntityView]]:
        grouped: dict[str, list[_EntityView]] = {}
        for entity in self.entities:
            grouped.setdefault(str(entity.node.properties[property_name]), []).append(entity)
        return grouped

    def _index_owner_intervals(self) -> dict[str, list[_OwnerInterval]]:
        grouped: dict[str, list[_OwnerInterval]] = {}
        for entity in self.entities:
            anchor = entity.definition_anchor or entity.body_anchor
            if anchor is None:
                continue
            grouped.setdefault(entity.file_path, []).append(
                _OwnerInterval(
                    entity_id=entity.node.id,
                    start_line=int(anchor.properties["start_line"]),
                    start_col=int(anchor.properties["start_col"]),
                    end_line=int(anchor.properties["end_line"]),
                    end_col=int(anchor.properties["end_col"]),
                )
            )
        for intervals in grouped.values():
            intervals.sort(key=lambda interval: _span_size(interval), reverse=False)
        return grouped


def _find_parent_file_id(entity_id: str, parent_by_child: dict[str, str], file_path_by_file_id: dict[str, str]) -> str | None:
    parent_id = parent_by_child.get(entity_id)
    while parent_id is not None:
        if parent_id in file_path_by_file_id:
            return parent_id
        parent_id = parent_by_child.get(parent_id)
    return None


def _location_position(location: JsonObject) -> tuple[str, int, int]:
    file_path = _normalize_path(str(location["relativePath"]))
    range_payload = location["range"]
    assert isinstance(range_payload, dict)
    start = range_payload["start"]
    assert isinstance(start, dict)
    return file_path, int(start["line"]), int(start["character"])


def _anchor_contains(anchor: GraphNode, line: int, column: int) -> bool:
    return _span_contains(
        int(anchor.properties["start_line"]),
        int(anchor.properties["start_col"]),
        int(anchor.properties["end_line"]),
        int(anchor.properties["end_col"]),
        line,
        column,
    )


def _span_contains(start_line: int, start_col: int, end_line: int, end_col: int, line: int, column: int) -> bool:
    if (line, column) < (start_line, start_col):
        return False
    return (line, column) <= (end_line, end_col)


def _span_size(interval: _OwnerInterval) -> tuple[int, int]:
    return interval.end_line - interval.start_line, interval.end_col - interval.start_col


def _read_package_name(root_node: Node, source_bytes: bytes) -> str:
    for child in root_node.named_children:
        if child.type != "package_declaration":
            continue
        package_node = next((node for node in child.named_children if node.type in {"identifier", "scoped_identifier"}), None)
        return _node_text(package_node, source_bytes) if package_node is not None else ""
    return ""


def _read_imports(root_node: Node, source_bytes: bytes) -> dict[str, str]:
    imports: dict[str, str] = {}
    for child in root_node.named_children:
        if child.type != "import_declaration":
            continue
        import_node = next((node for node in child.named_children if node.type in {"identifier", "scoped_identifier"}), None)
        if import_node is None:
            continue
        qualified_name = _node_text(import_node, source_bytes)
        imports[_simple_name(qualified_name)] = qualified_name
    return imports


def _find_import_node(root_node: Node, source_bytes: bytes, imported_name: str) -> Node | None:
    for child in root_node.named_children:
        if child.type != "import_declaration":
            continue
        import_node = next((node for node in child.named_children if node.type in {"identifier", "scoped_identifier"}), None)
        if import_node is not None and _node_text(import_node, source_bytes) == imported_name:
            return import_node
    return None


def _type_fallback_names(type_name: str, package_name: str, imports: dict[str, str]) -> tuple[str, ...]:
    cleaned_name = type_name.split("<", 1)[0].replace("[]", "").strip()
    simple_name = _simple_name(cleaned_name)
    names = [cleaned_name]
    if simple_name in imports:
        names.append(imports[simple_name])
    if package_name and "." not in cleaned_name:
        names.append(f"{package_name}.{cleaned_name}")
    names.append(simple_name)
    return tuple(dict.fromkeys(name for name in names if name))


def _is_top_level_entity(entity: _EntityView) -> bool:
    qualified_name = str(entity.node.properties["qualified_name"])
    return "#" not in qualified_name and qualified_name.count(".") <= 2


def _split_method_owner_and_name(qualified_name: str) -> tuple[str, str]:
    owner_name, _, method_signature = qualified_name.partition("#")
    method_name = method_signature.split("(", 1)[0]
    return owner_name, method_name


def _simple_name(qualified_name: str) -> str:
    return qualified_name.rsplit(".", 1)[-1]


def _last_named_child(node: Node, child_type: str) -> Node | None:
    return next((child for child in reversed(node.named_children) if child.type == child_type), None)


def _has_parent(node: Node, parent_type: str) -> bool:
    return node.parent is not None and node.parent.type == parent_type


def _has_ancestor(node: Node, ancestor_types: set[str]) -> bool:
    parent = node.parent
    while parent is not None:
        if parent.type in ancestor_types:
            return True
        if parent.type in DECLARATION_NODE_KINDS:
            return False
        parent = parent.parent
    return False


def _node_text(node: Node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte : node.end_byte].decode("utf-8")


def _normalize_path(path: str) -> str:
    return PurePosixPath(path).as_posix()
