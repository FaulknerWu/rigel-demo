"""基于 Tree-sitter 的 Java 物理结构解析器。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import PurePosixPath

import tree_sitter_java
from tree_sitter import Language, Node, Parser

from rigel_demo.graph_ir import Anchor, EdgeType, Entity, File, GraphEdge, GraphIR, Module, Repository


JAVA_LANGUAGE = Language(tree_sitter_java.language())

# 这些常量会进入图谱属性或 ID，集中定义可避免解析器和测试之间出现隐式约定。
TREE_SITTER_PROVENANCE = "tree-sitter"
PHYSICAL_MEMBERSHIP_KIND = "physical-membership"
DEFAULT_MODULE_NAME = "root"
DEFAULT_MODULE_ECOSYSTEM = "maven"
DEFAULT_ZONE = "prod"
HASH_PREFIX = "sha256:"

TYPE_DECLARATION_KINDS = {
    "annotation_type_declaration": "interface",
    "class_declaration": "class",
    "enum_declaration": "enum",
    "interface_declaration": "interface",
    "record_declaration": "class",
}

# Tree-sitter Java 会把构造函数、普通方法、注解元素拆成不同节点；图谱层先归一为 method。
METHOD_DECLARATION_KINDS = {
    "annotation_type_element_declaration",
    "compact_constructor_declaration",
    "constructor_declaration",
    "method_declaration",
}

# 只有进入这些节点时才继续向下寻找子实体，避免把表达式内部的普通标识符误识别为实体。
BODY_NODE_KINDS = {
    "annotation_type_body",
    "block",
    "class_body",
    "constructor_body",
    "enum_body",
    "interface_body",
}


@dataclass(frozen=True, slots=True)
class JavaParseRequest:
    """Java 文件解析请求。

    请求对象显式携带仓库和模块上下文，因为单文件源码本身无法可靠反推出
    所属模块、生态和运行分区。
    """

    repository_name: str
    module_name: str = DEFAULT_MODULE_NAME
    module_root_path: str = "."
    module_ecosystem: str = DEFAULT_MODULE_ECOSYSTEM
    zone: str = DEFAULT_ZONE


@dataclass(slots=True)
class _EntityRecord:
    """实体构建过程中的临时树节点。

    先构建记录树再写入 GraphIR，可以让父子实体的包含关系和锚点生成保持在
    同一个递归出口，减少不同实体类型之间的重复逻辑。
    """

    entity: Entity
    declaration_node: Node
    body_node: Node | None
    child_records: list["_EntityRecord"] = field(default_factory=list)


def parse_java_file(
    source: str | bytes,
    relative_path: str,
    *,
    request: JavaParseRequest,
) -> GraphIR:
    """解析单个 Java 文件，生成文件、实体、锚点与基础包含关系。"""

    source_bytes = source.encode("utf-8") if isinstance(source, str) else source
    parser = Parser()
    parser.language = JAVA_LANGUAGE
    syntax_tree = parser.parse(source_bytes)
    package_name = _read_package_name(syntax_tree.root_node, source_bytes)

    repository = Repository(repo_id=f"repo:{request.repository_name}", name=request.repository_name)
    module_id = f"module:{request.repository_name}:{request.module_name}"
    module = Module(
        module_id=module_id,
        name=request.module_name,
        root_path=request.module_root_path,
        ecosystem=request.module_ecosystem,
        zone=request.zone,
    )
    normalized_path = _normalize_path(relative_path)
    file_model = File(
        file_id=f"file:{request.repository_name}:{normalized_path}",
        relative_path=normalized_path,
        language="java",
        zone=request.zone,
        content_hash=_content_hash(source_bytes),
    )

    graph = GraphIR()
    graph.add_node(repository)
    graph.add_node(module)
    graph.add_node(file_model)
    _add_contains_edge(graph, repository.repo_id, module.module_id)
    _add_contains_edge(graph, module.module_id, file_model.file_id)
    _add_anchor(graph, file_model.file_id, "definition", syntax_tree.root_node)
    _add_anchor(graph, file_model.file_id, "body", syntax_tree.root_node)

    top_level_records = _extract_entity_records(
        syntax_tree.root_node,
        source_bytes,
        package_name=package_name,
        file_id=file_model.file_id,
        parent_qualified_name=package_name,
    )
    for record in top_level_records:
        _emit_entity_record(graph, file_model.file_id, record)

    return graph


def _extract_entity_records(
    node: Node,
    source_bytes: bytes,
    *,
    package_name: str,
    file_id: str,
    parent_qualified_name: str,
) -> list[_EntityRecord]:
    """从 AST 子树提取当前层级下的代码实体。

    parent_qualified_name 表示当前语义容器，用于区分顶层类型、嵌套类型、方法内
    局部类型以及成员字段，保证同名实体在不同作用域下拥有不同 qualified_name。
    """

    records: list[_EntityRecord] = []
    for child in node.named_children:
        if child.type in TYPE_DECLARATION_KINDS:
            record = _create_type_record(child, source_bytes, package_name, file_id, parent_qualified_name)
            records.append(record)
            continue
        if child.type in METHOD_DECLARATION_KINDS:
            record = _create_method_record(child, source_bytes, package_name, file_id, parent_qualified_name)
            records.append(record)
            continue
        if child.type == "field_declaration":
            records.extend(_create_field_records(child, source_bytes, package_name, file_id, parent_qualified_name))
            continue

        if child.type in BODY_NODE_KINDS or node.type == "program":
            records.extend(
                _extract_entity_records(
                    child,
                    source_bytes,
                    package_name=package_name,
                    file_id=file_id,
                    parent_qualified_name=parent_qualified_name,
                )
            )
    return records


def _create_type_record(
    node: Node,
    source_bytes: bytes,
    package_name: str,
    file_id: str,
    parent_qualified_name: str,
) -> _EntityRecord:
    """创建类型实体记录，并递归收集其直接成员。"""

    name_node = _required_name_node(node)
    display_name = _node_text(name_node, source_bytes)
    qualified_name = _join_qualified_name(parent_qualified_name, display_name)
    body_node = _first_child_with_type(node, BODY_NODE_KINDS)
    record = _EntityRecord(
        entity=_entity(
            file_id=file_id,
            package_name=package_name,
            display_name=display_name,
            qualified_name=qualified_name,
            kind_norm=TYPE_DECLARATION_KINDS[node.type],
            kind_raw=node.type,
            semantic_source=_semantic_source(node, source_bytes),
        ),
        declaration_node=node,
        body_node=body_node,
    )
    if body_node is not None:
        record.child_records.extend(
            _extract_entity_records(
                body_node,
                source_bytes,
                package_name=package_name,
                file_id=file_id,
                parent_qualified_name=qualified_name,
            )
        )
    return record


def _create_method_record(
    node: Node,
    source_bytes: bytes,
    package_name: str,
    file_id: str,
    parent_qualified_name: str,
) -> _EntityRecord:
    """创建方法实体记录。

    qualified_name 中包含参数类型签名，目的是让 Java 重载方法在同一父级下保持
    稳定可区分。
    """

    name_node = _required_name_node(node)
    display_name = _node_text(name_node, source_bytes)
    qualified_name = f"{parent_qualified_name}#{display_name}{_parameter_signature(node, source_bytes)}"
    body_node = _first_child_with_type(node, BODY_NODE_KINDS)
    record = _EntityRecord(
        entity=_entity(
            file_id=file_id,
            package_name=package_name,
            display_name=display_name,
            qualified_name=qualified_name,
            kind_norm="method",
            kind_raw=node.type,
            semantic_source=_semantic_source(node, source_bytes),
        ),
        declaration_node=node,
        body_node=body_node,
    )
    if body_node is not None:
        record.child_records.extend(
            _extract_entity_records(
                body_node,
                source_bytes,
                package_name=package_name,
                file_id=file_id,
                parent_qualified_name=qualified_name,
            )
        )
    return record


def _create_field_records(
    node: Node,
    source_bytes: bytes,
    package_name: str,
    file_id: str,
    parent_qualified_name: str,
) -> list[_EntityRecord]:
    """从字段声明中拆分出每一个变量实体。

    Java 允许 `int a, b` 这种多变量声明；图谱检索通常关心单个字段，因此这里
    保留共享声明锚点，同时为每个 declarator 生成独立实体。
    """

    records: list[_EntityRecord] = []
    for declarator in node.named_children:
        if declarator.type != "variable_declarator":
            continue
        name_node = _required_name_node(declarator)
        display_name = _node_text(name_node, source_bytes)
        qualified_name = f"{parent_qualified_name}#{display_name}"
        records.append(
            _EntityRecord(
                entity=_entity(
                    file_id=file_id,
                    package_name=package_name,
                    display_name=display_name,
                    qualified_name=qualified_name,
                    kind_norm="field",
                    kind_raw=node.type,
                    semantic_source=_semantic_source(node, source_bytes, extra=display_name),
                ),
                declaration_node=node,
                body_node=declarator,
            )
        )
    return records


def _emit_entity_record(graph: GraphIR, parent_id: str, record: _EntityRecord) -> None:
    """把临时实体树写入 GraphIR。"""

    graph.add_node(record.entity)
    _add_contains_edge(graph, parent_id, record.entity.entity_id)
    _add_anchor(graph, record.entity.entity_id, "definition", record.declaration_node)
    if record.body_node is not None:
        _add_anchor(graph, record.entity.entity_id, "body", record.body_node)
    for child_record in record.child_records:
        _emit_entity_record(graph, record.entity.entity_id, child_record)


def _entity(
    *,
    file_id: str,
    package_name: str,
    display_name: str,
    qualified_name: str,
    kind_norm: str,
    kind_raw: str,
    semantic_source: bytes,
) -> Entity:
    """构造 Java 实体节点。

    entity_key 只依赖语言和 qualified_name；entity_id 再叠加 file_id，避免不同
    文件内相同限定名在物理图层发生碰撞。
    """

    entity_key = f"java:{qualified_name}"
    return Entity(
        entity_id=f"entity:{file_id}:{entity_key}",
        entity_key=entity_key,
        display_name=display_name,
        qualified_name=qualified_name or _join_qualified_name(package_name, display_name),
        kind_norm=kind_norm,
        kind_raw=kind_raw,
        origin="internal",
        semantic_hash=_content_hash(semantic_source),
    )


def _add_contains_edge(graph: GraphIR, source_id: str, target_id: str) -> None:
    """添加物理包含边。"""

    graph.add_edge(
        GraphEdge.create(
            EdgeType.CONTAINS,
            source_id,
            target_id,
            kind=PHYSICAL_MEMBERSHIP_KIND,
            provenance=TREE_SITTER_PROVENANCE,
            confidence=1.0,
        )
    )


def _add_anchor(graph: GraphIR, owner_id: str, role: str, node: Node) -> None:
    """为节点添加源码位置锚点。

    Tree-sitter 坐标从 0 开始；图谱输出改为 1 开始，便于和编辑器、诊断信息展示
    对齐。
    """

    anchor = Anchor(
        anchor_id=f"anchor:{owner_id}:{role}:{node.start_byte}:{node.end_byte}",
        start_line=node.start_point.row + 1,
        start_col=node.start_point.column + 1,
        end_line=node.end_point.row + 1,
        end_col=node.end_point.column + 1,
        role=role,
    )
    graph.add_node(anchor)
    graph.add_edge(
        GraphEdge.create(
            EdgeType.HAS_ANCHOR,
            owner_id,
            anchor.anchor_id,
            role=role,
            provenance=TREE_SITTER_PROVENANCE,
            confidence=1.0,
        )
    )


def _read_package_name(root_node: Node, source_bytes: bytes) -> str:
    """读取 Java package 声明；默认包返回空字符串。"""

    for child in root_node.named_children:
        if child.type != "package_declaration":
            continue
        for package_child in child.named_children:
            if package_child.type in {"identifier", "scoped_identifier"}:
                return _node_text(package_child, source_bytes)
    return ""


def _parameter_signature(node: Node, source_bytes: bytes) -> str:
    """生成方法参数类型签名。"""

    parameters = next((child for child in node.named_children if child.type == "formal_parameters"), None)
    if parameters is None:
        return "()"
    parameter_types = [_parameter_type(parameter, source_bytes) for parameter in parameters.named_children]
    return f"({','.join(parameter_type for parameter_type in parameter_types if parameter_type)})"


def _parameter_type(parameter_node: Node, source_bytes: bytes) -> str:
    """提取单个参数的类型文本。"""

    for child in parameter_node.named_children:
        if child.type != "identifier":
            return _node_text(child, source_bytes)
    return ""


def _semantic_source(node: Node, source_bytes: bytes, *, extra: str = "") -> bytes:
    """生成用于实体 semantic_hash 的规范化源码片段。"""

    tokens = list(_semantic_tokens(node, source_bytes))
    if extra:
        tokens.append(extra)
    return " ".join(tokens).encode("utf-8")


def _semantic_tokens(node: Node, source_bytes: bytes) -> list[str]:
    """递归提取语义 token，刻意忽略注释与格式空白。"""

    if node.is_extra or node.type in {"line_comment", "block_comment"}:
        return []
    if node.child_count == 0:
        text = _node_text(node, source_bytes).strip()
        return [text] if text else []
    tokens: list[str] = []
    for child in node.children:
        tokens.extend(_semantic_tokens(child, source_bytes))
    return tokens


def _first_child_with_type(node: Node, child_types: set[str]) -> Node | None:
    return next((child for child in node.named_children if child.type in child_types), None)


def _required_name_node(node: Node) -> Node:
    """读取 AST 节点名称，缺失时立即失败以暴露不支持的语法形态。"""

    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return name_node
    identifier = next((child for child in node.named_children if child.type == "identifier"), None)
    if identifier is None:
        raise ValueError(f"无法从 Java AST 节点提取名称：{node.type}")
    return identifier


def _node_text(node: Node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte : node.end_byte].decode("utf-8")


def _join_qualified_name(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def _normalize_path(path: str) -> str:
    return PurePosixPath(path).as_posix()


def _content_hash(content: bytes) -> str:
    return f"{HASH_PREFIX}{hashlib.sha256(content).hexdigest()}"
