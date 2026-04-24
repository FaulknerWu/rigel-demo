"""Rigel 图谱中间表示。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Literal, Self

JsonPrimitive = str | int | float | bool | None
JsonValue = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject = dict[str, JsonValue]


class NodeType(StrEnum):
    """图谱允许出现的节点类型。"""

    REPOSITORY = "Repository"
    MODULE = "Module"
    FILE = "File"
    ENTITY = "Entity"
    ANCHOR = "Anchor"
    SUMMARY = "Summary"


class EdgeType(StrEnum):
    """图谱允许出现的顶级边类型。"""

    CONTAINS = "CONTAINS"
    DEPENDS_ON = "DEPENDS_ON"
    SPECIALIZES = "SPECIALIZES"
    ALIASES = "ALIASES"
    HAS_ANCHOR = "HAS_ANCHOR"
    DESCRIBES = "DESCRIBES"


@dataclass(frozen=True, slots=True)
class GraphNode:
    """统一节点外壳，便于消费者按类型分发处理。"""

    id: str
    type: NodeType
    properties: JsonObject

    def to_json(self) -> JsonObject:
        return {
            "id": self.id,
            "type": self.type.value,
            "properties": self.properties,
        }


@dataclass(frozen=True, slots=True)
class Repository:
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
    entity_id: str
    entity_key: str
    display_name: str
    qualified_name: str
    kind_norm: str
    kind_raw: str
    origin: Literal["internal", "external", "generated"]
    semantic_hash: str

    def to_node(self) -> GraphNode:
        return GraphNode(
            id=self.entity_id,
            type=NodeType.ENTITY,
            properties=_dataclass_properties(self),
        )


@dataclass(frozen=True, slots=True)
class Anchor:
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
    summary_id: str
    text: str
    purpose: Literal["retrieval", "rollup"]
    source_hash: str
    embedding_model: str
    embedding: list[float]

    def to_node(self) -> GraphNode:
        return GraphNode(
            id=self.summary_id,
            type=NodeType.SUMMARY,
            properties=_dataclass_properties(self),
        )


@dataclass(frozen=True, slots=True)
class GraphEdge:
    """图谱边。

    关系语义细节必须写入 properties，例如 kind、provenance、confidence、role。
    顶层 type 只保留六类稳定关系族，避免把图模型绑定到特定语言或数据库。
    """

    id: str
    type: EdgeType
    source_id: str
    target_id: str
    properties: JsonObject = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        edge_type: EdgeType,
        source_id: str,
        target_id: str,
        *,
        kind: str | None = None,
        provenance: str | None = None,
        confidence: float | None = None,
        **properties: JsonValue,
    ) -> Self:
        edge_properties: JsonObject = dict(properties)
        if kind is not None:
            edge_properties["kind"] = kind
        if provenance is not None:
            edge_properties["provenance"] = provenance
        if confidence is not None:
            edge_properties["confidence"] = confidence

        return cls(
            id=_edge_id(edge_type, source_id, target_id, edge_properties),
            type=edge_type,
            source_id=source_id,
            target_id=target_id,
            properties=edge_properties,
        )

    def to_json(self) -> JsonObject:
        return {
            "id": self.id,
            "type": self.type.value,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "properties": self.properties,
        }


@dataclass(slots=True)
class GraphIR:
    """图谱中间表示根对象。"""

    schema_version: str = "rigel.graph-ir.v1"
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)

    def add_node(self, model: GraphNode | Repository | Module | File | Entity | Anchor | Summary) -> None:
        node = model if isinstance(model, GraphNode) else model.to_node()
        self.nodes.append(node)

    def add_edge(self, edge: GraphEdge) -> None:
        self.edges.append(edge)

    def to_json(self) -> JsonObject:
        return {
            "schema_version": self.schema_version,
            "nodes": [node.to_json() for node in self.nodes],
            "edges": [edge.to_json() for edge in self.edges],
        }


def _dataclass_properties(model: object) -> JsonObject:
    return {key: value for key, value in asdict(model).items()}


def _edge_id(edge_type: EdgeType, source_id: str, target_id: str, properties: JsonObject) -> str:
    semantic_suffix = properties.get("kind") or properties.get("role") or "default"
    return f"{edge_type.value}:{source_id}:{target_id}:{semantic_suffix}"
