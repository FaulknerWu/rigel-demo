"""Rigel 图谱中间表示。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Self, TYPE_CHECKING

if TYPE_CHECKING:
    from rigel_demo.entities import GraphDomainNode

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

    def __post_init__(self) -> None:
        _require_non_empty_string(self.id, "GraphNode.id")


@dataclass(frozen=True, slots=True)
class GraphEdge:
    """图谱边。"""

    id: str
    type: EdgeType
    source_id: str
    target_id: str
    properties: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty_string(self.id, "GraphEdge.id")
        _require_non_empty_string(self.source_id, "GraphEdge.source_id")
        _require_non_empty_string(self.target_id, "GraphEdge.target_id")
        _edge_semantic_suffix(self.properties)
        _validate_confidence(self.properties)

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
        """创建带稳定标识的边。

        边 ID 由类型、两端节点和关键语义后缀共同决定，避免同一对节点上不同
        语义关系互相覆盖。
        """

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


@dataclass(slots=True)
class GraphIR:
    """图谱中间表示根对象。"""

    schema_version: str = "rigel.graph-ir.v1"
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)

    def add_node(self, model: GraphNode | "GraphDomainNode") -> None:
        node = model if isinstance(model, GraphNode) else model.to_node()
        self.nodes.append(node)

    def add_edge(self, edge: GraphEdge) -> None:
        self.edges.append(edge)


def _edge_id(edge_type: EdgeType, source_id: str, target_id: str, properties: JsonObject) -> str:
    """生成可读且语义稳定的边 ID。"""

    semantic_suffix = _edge_semantic_suffix(properties)
    return f"{edge_type.value}:{source_id}:{target_id}:{semantic_suffix}"


def _edge_semantic_suffix(properties: JsonObject) -> str:
    for property_name in ("kind", "role"):
        value = properties.get(property_name)
        if value is None:
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()
        raise ValueError(f"GraphEdge.{property_name} 必须是非空字符串")
    raise ValueError("GraphEdge 必须包含非空 kind 或 role")


def _require_non_empty_string(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} 必须是非空字符串")


def _validate_confidence(properties: JsonObject) -> None:
    value = properties.get("confidence")
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("GraphEdge.confidence 必须是数字")
    if value < 0 or value > 1:
        raise ValueError("GraphEdge.confidence 必须在 0 到 1 之间")
