"""Java 源码 AST 通用读取工具。"""

from __future__ import annotations

from pathlib import PurePosixPath

from tree_sitter import Node


def read_package_name(root_node: Node, source_bytes: bytes) -> str:
    for child in root_node.named_children:
        if child.type != "package_declaration":
            continue
        package_node = _qualified_name_node(child)
        return node_text(package_node, source_bytes) if package_node is not None else ""
    return ""


def read_imports(root_node: Node, source_bytes: bytes) -> dict[str, str]:
    imports: dict[str, str] = {}
    for child in root_node.named_children:
        if child.type != "import_declaration":
            continue
        import_node = _qualified_name_node(child)
        if import_node is None:
            continue
        qualified_name = node_text(import_node, source_bytes)
        imports[simple_name(qualified_name)] = qualified_name
    return imports


def find_import_node(root_node: Node, source_bytes: bytes, imported_name: str) -> Node | None:
    for child in root_node.named_children:
        if child.type != "import_declaration":
            continue
        import_node = _qualified_name_node(child)
        if import_node is not None and node_text(import_node, source_bytes) == imported_name:
            return import_node
    return None


def simple_name(qualified_name: str) -> str:
    return qualified_name.rsplit(".", 1)[-1]


def last_named_child(node: Node, child_type: str) -> Node | None:
    return next((child for child in reversed(node.named_children) if child.type == child_type), None)


def has_parent(node: Node, parent_type: str) -> bool:
    return node.parent is not None and node.parent.type == parent_type


def has_ancestor_until_declaration(node: Node, ancestor_types: set[str], declaration_node_kinds: set[str]) -> bool:
    parent = node.parent
    while parent is not None:
        if parent.type in ancestor_types:
            return True
        if parent.type in declaration_node_kinds:
            return False
        parent = parent.parent
    return False


def node_text(node: Node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte : node.end_byte].decode("utf-8")


def _qualified_name_node(node: Node) -> Node | None:
    return next(
        (
            child
            for child in node.named_children
            if child.type in {"identifier", "scoped_identifier"}
        ),
        None,
    )


def normalize_path(path: str) -> str:
    return PurePosixPath(path).as_posix()
