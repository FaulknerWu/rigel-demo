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

    def __post_init__(self) -> None:
        _require_non_empty_string(self.repo_id, "Repository.repo_id")
        _require_non_empty_string(self.name, "Repository.name")

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

    def __post_init__(self) -> None:
        _require_non_empty_string(self.module_id, "Module.module_id")
        _require_non_empty_string(self.name, "Module.name")
        _require_non_empty_string(self.root_path, "Module.root_path")
        _require_non_empty_string(self.ecosystem, "Module.ecosystem")
        _require_non_empty_string(self.zone, "Module.zone")

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

    def __post_init__(self) -> None:
        _require_non_empty_string(self.file_id, "File.file_id")
        _require_non_empty_string(self.relative_path, "File.relative_path")
        _require_non_empty_string(self.language, "File.language")
        _require_non_empty_string(self.zone, "File.zone")
        _require_non_empty_string(self.content_hash, "File.content_hash")
        _require_non_empty_string(self.position_encoding, "File.position_encoding")

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

    def __post_init__(self) -> None:
        _require_non_empty_string(self.anchor_id, "Anchor.anchor_id")
        _require_positive_int(self.start_line, "Anchor.start_line")
        _require_positive_int(self.start_col, "Anchor.start_col")
        _require_positive_int(self.end_line, "Anchor.end_line")
        _require_positive_int(self.end_col, "Anchor.end_col")
        _require_non_empty_string(self.role, "Anchor.role")
        if (self.end_line, self.end_col) < (self.start_line, self.start_col):
            raise ValueError("Anchor 结束位置必须大于或等于开始位置")

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

    def __post_init__(self) -> None:
        _require_non_empty_string(self.summary_id, "Summary.summary_id")
        _require_non_empty_string(self.text, "Summary.text")
        _require_non_empty_string(self.source_hash, "Summary.source_hash")
        _require_non_empty_string(self.summary_model, "Summary.summary_model")
        _require_non_empty_string(self.embedding_model, "Summary.embedding_model")
        if self.purpose not in {"retrieval", "rollup"}:
            raise ValueError("Summary.purpose 必须是 retrieval 或 rollup")
        _require_positive_int(self.embedding_dimensions, "Summary.embedding_dimensions")
        _validate_embedding(self.embedding, self.embedding_dimensions)

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


def _require_positive_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} 必须是整数")
    if value <= 0:
        raise ValueError(f"{field_name} 必须大于 0")


def _validate_embedding(embedding: list[float], embedding_dimensions: int) -> None:
    if len(embedding) != embedding_dimensions:
        raise ValueError("Summary.embedding 长度必须等于 Summary.embedding_dimensions")
    for value in embedding:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("Summary.embedding 必须全部是数字")
