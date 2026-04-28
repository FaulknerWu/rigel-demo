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
    node_text,
    read_imports,
    read_package_name,
    simple_name,
)

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

class JavaLspClient(Protocol):
    """语义边补全依赖的最小 LSP 客户端接口。"""

    def request_definition(self, file_path: str, line: int, column: int) -> list[JsonObject]: ...

    def request_references(self, file_path: str, line: int, column: int) -> list[JsonObject]: ...

    def request_hover(self, relative_file_path: str, line: int, column: int) -> JsonObject | None: ...


@dataclass(frozen=True, slots=True)
class _SemanticCandidate:
    source_entity_id: str
    edge_type: EdgeType
    kind: str
    file_path: str
    line: int
    column: int
    fallback_names: tuple[str, ...]


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
    graph_index = GraphIndex(graph)
    lsp_context = _started_lsp(repository_root, request.lsp_timeout_seconds) if lsp_client is None else nullcontext(lsp_client)

    with lsp_context as active_lsp:
        # 候选边先由 Tree-sitter 定位语法位置，再交给 LSP 解析真实目标，兼顾覆盖率和语义精度。
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

            # LSP 在未完整编译或依赖缺失时可能定位失败，名称回退保留可用但低置信度的演示图谱。
            fallback_target = graph_index.find_unique_entity_by_names(candidate.fallback_names)
            if fallback_target is not None and fallback_target.node.id != candidate.source_entity_id:
                _add_semantic_edge(
                    graph,
                    candidate,
                    fallback_target.node.id,
                    provenance=TREE_SITTER_PROVENANCE,
                    confidence=TREE_SITTER_CONFIDENCE,
                )

        # references 与 override 依赖完整实体索引，放在候选边补全后统一追加，避免重复扫描 AST。
        _add_lsp_reference_edges(graph, active_lsp, graph_index)
        _add_override_edges(graph, graph_index)

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
        if not source_path.exists():
            continue
        source_bytes = source_path.read_bytes()
        root_node = parser.parse(source_bytes).root_node
        package_name = read_package_name(root_node, source_bytes)
        imports = read_imports(root_node, source_bytes)
        file_entities = graph_index.entities_by_file_path.get(file_path, [])
        top_level_entities = [entity for entity in file_entities if _is_top_level_entity(entity)]

        # import 属于文件级语义，这里挂到顶层实体上，避免把依赖关系散落到无源码实体的文件节点。
        for imported_name in imports.values():
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
    graph_index: GraphIndex,
    file_path: str,
    package_name: str,
    imports: dict[str, str],
    candidates: list[_SemanticCandidate],
) -> None:
    owner = graph_index.find_owner_entity(file_path, node.start_point.row + 1, node.start_point.column + 1)
    owner_id = owner.node.id if owner is not None else None

    if owner_id is not None and node.type == "method_invocation":
        name_node = node.child_by_field_name("name") or last_named_child(node, "identifier")
        if name_node is not None:
            method_name = node_text(name_node, source_bytes)
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

    # 继承/实现会在专门分支生成 SPECIALIZES 边，这里排除这些容器以免同一类型同时产生依赖边。
    if owner_id is not None and node.type in TYPE_REFERENCE_NODE_KINDS and not has_ancestor_until_declaration(node, INHERITANCE_CONTAINER_KINDS, DECLARATION_NODE_KINDS):
        type_name = node_text(node, source_bytes)
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

    if owner_id is not None and has_parent(node, "superclass") and node.type in TYPE_REFERENCE_NODE_KINDS:
        type_name = node_text(node, source_bytes)
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

    if owner_id is not None and has_ancestor_until_declaration(node, {"super_interfaces", "extends_interfaces"}, DECLARATION_NODE_KINDS) and node.type in TYPE_REFERENCE_NODE_KINDS:
        type_name = node_text(node, source_bytes)
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


def _add_lsp_reference_edges(graph: GraphIR, lsp_client: JavaLspClient, graph_index: GraphIndex) -> None:
    for target_entity in graph_index.entities:
        anchor = target_entity.definition_anchor
        if anchor is None:
            continue
        # GraphIR 对外使用 1 基坐标；LSP 协议使用 0 基坐标，请求前必须还原。
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
    graph_index: GraphIndex,
    candidate: _SemanticCandidate,
) -> EntityView | None:
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
        # 语义增强不能因为单个 LSP 请求失败中断整个索引流程，失败位置交给名称回退处理。
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


def _type_fallback_names(type_name: str, package_name: str, imports: dict[str, str]) -> tuple[str, ...]:
    cleaned_name = type_name.split("<", 1)[0].replace("[]", "").strip()
    cleaned_simple_name = simple_name(cleaned_name)
    names = [cleaned_name]
    # 回退名称按“源码写法 -> import 展开 -> 同包限定名 -> 简名”排列，优先保留更精确身份。
    if cleaned_simple_name in imports:
        names.append(imports[cleaned_simple_name])
    if package_name and "." not in cleaned_name:
        names.append(f"{package_name}.{cleaned_name}")
    names.append(cleaned_simple_name)
    return tuple(dict.fromkeys(name for name in names if name))


def _is_top_level_entity(entity: EntityView) -> bool:
    qualified_name = str(entity.node.properties["qualified_name"])
    return "#" not in qualified_name and qualified_name.count(".") <= 2


def _split_method_owner_and_name(qualified_name: str) -> tuple[str, str]:
    owner_name, _, method_signature = qualified_name.partition("#")
    method_name = method_signature.split("(", 1)[0]
    return owner_name, method_name

