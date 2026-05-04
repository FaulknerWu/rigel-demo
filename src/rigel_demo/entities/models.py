"""Rigel 图节点领域模型。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from rigel_demo.graph.ir import GraphNode, JsonObject, NodeType


@dataclass(frozen=True, slots=True)
class Repository:
    """代码仓库节点，作为图谱物理层级的根。"""

    repo_id: str
    name: str

    def to_node(self) -> GraphNode:
        return GraphNode(
            id=self.repo_id,
            type=NodeType.REPOSITORY,
            properties={"repo_id": self.repo_id, "name": self.name},
        )


@dataclass(frozen=True, slots=True)
class Module:
    """仓库内的构建或发布单元。"""

    module_id: str
    name: str
    root_path: str
    ecosystem: str
    zone: str

    def to_node(self) -> GraphNode:
        return GraphNode(
            id=self.module_id,
            type=NodeType.MODULE,
            properties=_dataclass_properties(self),
        )


@dataclass(frozen=True, slots=True)
class File:
    """源码文件节点。"""

    file_id: str
    relative_path: str
    language: str
    zone: str
    content_hash: str
    position_encoding: str = "UTF-8"

    def to_node(self) -> GraphNode:
        return GraphNode(
            id=self.file_id,
            type=NodeType.FILE,
            properties=_dataclass_properties(self),
        )


@dataclass(frozen=True, slots=True)
class Entity:
    """代码实体节点，例如类型、方法、字段等可检索的语义单元。"""

    entity_id: str
    entity_key: str
    display_name: str
    qualified_name: str
    kind_norm: str
    kind_raw: str
    origin: Literal["internal", "external", "generated"]
    semantic_hash: str

    def __post_init__(self) -> None:
        _require_non_empty_string(self.entity_id, "Entity.entity_id")
        _require_non_empty_string(self.entity_key, "Entity.entity_key")
        _require_non_empty_string(self.display_name, "Entity.display_name")
        _require_non_empty_string(self.qualified_name, "Entity.qualified_name")
        _require_non_empty_string(self.kind_norm, "Entity.kind_norm")
        _require_non_empty_string(self.kind_raw, "Entity.kind_raw")
        _require_non_empty_string(self.semantic_hash, "Entity.semantic_hash")
        if self.origin not in {"internal", "external", "generated"}:
            raise ValueError("Entity.origin 必须是 internal、external 或 generated")

    def to_node(self) -> GraphNode:
        return GraphNode(
            id=self.entity_id,
            type=NodeType.ENTITY,
            properties=_dataclass_properties(self),
        )


@dataclass(frozen=True, slots=True)
class Anchor:
    """实体或文件在源码中的位置锚点。"""

    anchor_id: str
    start_line: int
    start_col: int
    end_line: int
    end_col: int
    role: str

    def to_node(self) -> GraphNode:
        return GraphNode(
            id=self.anchor_id,
            type=NodeType.ANCHOR,
            properties=_dataclass_properties(self),
        )


@dataclass(frozen=True, slots=True)
class Summary:
    """面向检索或聚合展示的摘要节点。"""

    summary_id: str
    text: str
    purpose: Literal["retrieval", "rollup"]
    source_hash: str
    summary_model: str
    embedding_model: str
    embedding_dimensions: int
    embedding: list[float]

    def to_node(self) -> GraphNode:
        return GraphNode(
            id=self.summary_id,
            type=NodeType.SUMMARY,
            properties=_dataclass_properties(self),
        )


GraphDomainNode = Repository | Module | File | Entity | Anchor | Summary


def _dataclass_properties(model: object) -> JsonObject:
    return {key: value for key, value in asdict(model).items()}


def _require_non_empty_string(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} 必须是非空字符串")
