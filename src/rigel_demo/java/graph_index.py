"""Java 语义补全使用的图谱索引。"""

from __future__ import annotations

from dataclasses import dataclass

from rigel_demo.core.graph_ir import EdgeType, GraphIR, GraphNode, JsonObject, NodeType
from rigel_demo.java.source_utils import normalize_path


@dataclass(frozen=True, slots=True)
class EntityView:
    """语义补全阶段使用的实体视图，预绑定文件路径和常用锚点。"""

    node: GraphNode
    file_path: str
    name_anchor: GraphNode | None
    definition_anchor: GraphNode
    body_anchor: GraphNode | None


@dataclass(frozen=True, slots=True)
class OwnerInterval:
    """实体声明范围，用于把 LSP 返回的位置归属到最内层实体。"""

    entity_id: str
    start_line: int
    start_col: int
    end_line: int
    end_col: int


class GraphIndex:
    """为 GraphIR 建立只读索引，支撑 LSP 位置和图谱实体之间的相互映射。"""

    def __init__(self, graph: GraphIR) -> None:
        self.nodes_by_id = {node.id: node for node in graph.nodes}
        self.file_path_by_file_id = {
            node.id: str(node.properties["relative_path"])
            for node in graph.nodes
            if node.type == NodeType.FILE and node.properties["language"] == "java"
        }
        self.java_file_paths = list(self.file_path_by_file_id.values())
        self.anchor_by_id = {node.id: node for node in graph.nodes if node.type == NodeType.ANCHOR}
        self.anchors_by_owner_and_role = self._index_anchors(graph)
        self.entities = self._index_entities(graph)
        self.entities_by_file_path = self._group_entities_by_file_path()
        self.entities_by_qualified_name = self._group_entities_by_property("qualified_name")
        self.owner_intervals_by_file = self._index_owner_intervals()

    def find_unique_entity_by_qualified_name(self, qualified_name: str) -> EntityView | None:
        entities = self.entities_by_qualified_name.get(qualified_name, [])
        return entities[0] if len(entities) == 1 else None

    def find_location_target(self, location: JsonObject) -> EntityView | None:
        """把 LSP 定义位置解析为目标实体。"""

        file_path, line, column = location_position(location)
        containing_entities: list[tuple[EntityView, tuple[int, int]]] = []
        for entity in self.entities_by_file_path.get(file_path, []):
            anchor = entity.definition_anchor
            if anchor_contains(anchor, line + 1, column + 1):
                interval = OwnerInterval(
                    entity_id=entity.node.id,
                    start_line=int(anchor.properties["start_line"]),
                    start_col=int(anchor.properties["start_col"]),
                    end_line=int(anchor.properties["end_line"]),
                    end_col=int(anchor.properties["end_col"]),
                )
                containing_entities.append((entity, span_size(interval)))
        if containing_entities:
            # 定义跳转可能落在嵌套实体范围内，取最小声明区间可以定位到最具体目标。
            containing_entities.sort(key=lambda item: item[1])
            return containing_entities[0][0]
        return self.find_owner_entity(file_path, line + 1, column + 1)

    def find_location_owner(self, location: JsonObject) -> EntityView | None:
        """把 LSP 引用位置解析为拥有该引用的源码实体。"""

        file_path, line, column = location_position(location)
        return self.find_owner_entity(file_path, line + 1, column + 1)

    def find_owner_entity(self, file_path: str, line: int, column: int) -> EntityView | None:
        intervals = self.owner_intervals_by_file.get(normalize_path(file_path), [])
        for interval in intervals:
            if span_contains(interval.start_line, interval.start_col, interval.end_line, interval.end_col, line, column):
                return self._entity_by_id(interval.entity_id)
        return None

    def _entity_by_id(self, entity_id: str) -> EntityView | None:
        return next((entity for entity in self.entities if entity.node.id == entity_id), None)

    def _index_anchors(self, graph: GraphIR) -> dict[tuple[str, str], GraphNode]:
        anchors: dict[tuple[str, str], GraphNode] = {}
        for edge in graph.edges:
            if edge.type != EdgeType.HAS_ANCHOR:
                continue
            role = str(edge.properties["role"])
            anchors[(edge.source_id, role)] = self.anchor_by_id[edge.target_id]
        return anchors

    def _index_entities(self, graph: GraphIR) -> list[EntityView]:
        entities: list[EntityView] = []
        parent_by_child = {
            edge.target_id: edge.source_id
            for edge in graph.edges
            if edge.type == EdgeType.CONTAINS and edge.target_id in self.nodes_by_id
        }
        for node in graph.nodes:
            if node.type != NodeType.ENTITY:
                continue
            file_id = find_parent_file_id(node.id, parent_by_child, self.file_path_by_file_id)
            entities.append(
                EntityView(
                    node=node,
                    file_path=self.file_path_by_file_id[file_id],
                    name_anchor=self.anchors_by_owner_and_role.get((node.id, "name")),
                    definition_anchor=self.anchors_by_owner_and_role[(node.id, "definition")],
                    body_anchor=self.anchors_by_owner_and_role.get((node.id, "body")),
                )
            )
        return entities

    def _group_entities_by_file_path(self) -> dict[str, list[EntityView]]:
        grouped: dict[str, list[EntityView]] = {}
        for entity in self.entities:
            grouped.setdefault(entity.file_path, []).append(entity)
        return grouped

    def _group_entities_by_property(self, property_name: str) -> dict[str, list[EntityView]]:
        grouped: dict[str, list[EntityView]] = {}
        for entity in self.entities:
            grouped.setdefault(str(entity.node.properties[property_name]), []).append(entity)
        return grouped

    def _index_owner_intervals(self) -> dict[str, list[OwnerInterval]]:
        grouped: dict[str, list[OwnerInterval]] = {}
        for entity in self.entities:
            anchor = entity.definition_anchor
            grouped.setdefault(entity.file_path, []).append(
                OwnerInterval(
                    entity_id=entity.node.id,
                    start_line=int(anchor.properties["start_line"]),
                    start_col=int(anchor.properties["start_col"]),
                    end_line=int(anchor.properties["end_line"]),
                    end_col=int(anchor.properties["end_col"]),
                )
            )
        for intervals in grouped.values():
            # 小范围实体排在前面，确保同一位置优先归属到方法或局部类型而不是外层类型。
            intervals.sort(key=lambda interval: span_size(interval), reverse=False)
        return grouped


def find_parent_file_id(entity_id: str, parent_by_child: dict[str, str], file_path_by_file_id: dict[str, str]) -> str:
    parent_id = parent_by_child[entity_id]
    while parent_id not in file_path_by_file_id:
        parent_id = parent_by_child[parent_id]
    return parent_id


def location_position(location: JsonObject) -> tuple[str, int, int]:
    file_path = normalize_path(str(location["relativePath"]))
    range_payload = location["range"]
    assert isinstance(range_payload, dict)
    start = range_payload["start"]
    assert isinstance(start, dict)
    return file_path, int(start["line"]), int(start["character"])


def anchor_contains(anchor: GraphNode, line: int, column: int) -> bool:
    return span_contains(
        int(anchor.properties["start_line"]),
        int(anchor.properties["start_col"]),
        int(anchor.properties["end_line"]),
        int(anchor.properties["end_col"]),
        line,
        column,
    )


def span_contains(start_line: int, start_col: int, end_line: int, end_col: int, line: int, column: int) -> bool:
    if (line, column) < (start_line, start_col):
        return False
    return (line, column) <= (end_line, end_col)


def span_size(interval: OwnerInterval) -> tuple[int, int]:
    return interval.end_line - interval.start_line, interval.end_col - interval.start_col
