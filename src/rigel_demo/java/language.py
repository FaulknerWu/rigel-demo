"""Java Tree-sitter 语言配置与共享解析常量。"""

from __future__ import annotations

import tree_sitter_java
from tree_sitter import Language

JAVA_LANGUAGE = Language(tree_sitter_java.language())

# 这些常量会进入图谱属性或 ID，集中定义可避免解析器和测试之间出现隐式约定。
TREE_SITTER_PROVENANCE = "tree-sitter"

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
