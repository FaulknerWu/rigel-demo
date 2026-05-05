"""Web 与 GraphRAG 辅助查询共用的图谱证据读取服务。"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

from rigel_demo.project.summaries import RETRIEVAL_SUMMARY_PURPOSE
from rigel_demo.query.presentation import (
    anchor_sort_key,
    format_anchor,
    format_edge,
    format_graph_relation,
    format_node,
    format_source_file,
)
from rigel_demo.query.source_reader import (
    RepositorySourceReader,
    SourceFileNotFoundError,
    SourceLineRangeError,
    SourcePathError,
    SourceReadError,
)
from rigel_demo.storage.falkordb.store import FalkorDBConfig, FalkorDBStore

VISIBLE_NODE_TYPES = ("Repository", "Module", "File", "Entity")
VISIBLE_EDGE_TYPES = ("CONTAINS", "DEPENDS_ON", "SPECIALIZES", "ALIASES")

GraphExpansionDirection = Literal["incoming", "outgoing", "both"]


class RigelGraphReader:
    """读取 FalkorDBLite 中的 Rigel 图谱数据。"""

    def __init__(self, *, database_path: Path, graph_name: str) -> None:
        self._database_path = database_path
        self._graph_name = graph_name
        self._store = FalkorDBStore.connect(
            FalkorDBConfig(graph_name=self._graph_name, database_path=str(self._database_path))
        )

    def close(self) -> None:
        self._store.close()

    def __enter__(self) -> "RigelGraphReader":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def summary(self) -> dict[str, object]:
        """统计默认可视化语义图谱的节点、边与节点类型分布。"""

        node_count = self._scalar_query(
            """
            MATCH (node:RigelNode)
            WHERE node.rigel_type IN $visible_node_types
            RETURN count(node)
            """,
            {"visible_node_types": list(VISIBLE_NODE_TYPES)},
        )
        edge_count = self._scalar_query(
            """
            MATCH (source:RigelNode)-[edge]->(target:RigelNode)
            WHERE source.rigel_type IN $visible_node_types
              AND target.rigel_type IN $visible_node_types
              AND type(edge) IN $visible_edge_types
            RETURN count(edge)
            """,
            {
                "visible_node_types": list(VISIBLE_NODE_TYPES),
                "visible_edge_types": list(VISIBLE_EDGE_TYPES),
            },
        )
        type_rows = self._query(
            """
            MATCH (node:RigelNode)
            WHERE node.rigel_type IN $visible_node_types
            RETURN node.rigel_type, count(node)
            ORDER BY count(node) DESC
            """,
            {"visible_node_types": list(VISIBLE_NODE_TYPES)},
        )
        return {
            "node_count": node_count,
            "edge_count": edge_count,
            "node_types": [
                {"type": node_type, "count": count}
                for node_type, count in type_rows
            ],
        }

    def graph(self, *, limit: int | None = None) -> dict[str, list[dict[str, object]]]:
        """读取默认可视化语义节点和这些节点之间的语义边。"""

        node_rows = self._query(
            """
            MATCH (node:RigelNode)
            WHERE node.rigel_type IN $visible_node_types
            RETURN node.id, properties(node)
            """,
            {"visible_node_types": list(VISIBLE_NODE_TYPES)},
        )
        node_rows = _limit_rows(sorted(node_rows, key=_visual_node_sort_key), limit)
        nodes = [format_node(node_id, properties) for node_id, properties in node_rows]
        node_ids = [node["id"] for node in nodes]
        if not node_ids:
            return {"nodes": [], "edges": []}
        summaries_by_node_id = self._retrieval_summaries_by_node_id(node_ids)
        nodes = [
            _with_generated_summary(node, summaries_by_node_id)
            for node in nodes
        ]

        # 边只返回当前节点窗口内部的关系，避免前端收到指向缺失节点的悬空连线。
        edge_rows = self._query(
            """
            MATCH (source:RigelNode)-[edge]->(target:RigelNode)
            WHERE source.id IN $node_ids AND target.id IN $node_ids
              AND type(edge) IN $visible_edge_types
            RETURN source.id, target.id, type(edge), properties(edge)
            ORDER BY CASE type(edge) WHEN 'CONTAINS' THEN 0 ELSE 1 END, source.id, target.id
            """,
            {
                "node_ids": node_ids,
                "visible_edge_types": list(VISIBLE_EDGE_TYPES),
            },
        )
        edge_rows = _limit_rows(edge_rows, limit * 2 if limit is not None else None)
        edges = [format_edge(source_id, target_id, edge_type, properties) for source_id, target_id, edge_type, properties in edge_rows]
        return {"nodes": nodes, "edges": edges}

    def _retrieval_summaries_by_node_id(self, node_ids: list[object]) -> dict[str, str]:
        summary_rows = self._query(
            """
            MATCH (summary:RigelNode:Summary)-[:DESCRIBES]->(target:RigelNode)
            WHERE target.id IN $node_ids
              AND summary.purpose = $purpose
            RETURN target.id, summary.text
            ORDER BY target.id, summary.id
            """,
            {
                "node_ids": node_ids,
                "purpose": RETRIEVAL_SUMMARY_PURPOSE,
            },
        )

        summaries_by_node_id: dict[str, str] = {}
        for node_id, summary_text in summary_rows:
            node_id_text = str(node_id)
            if node_id_text not in summaries_by_node_id and isinstance(summary_text, str):
                summaries_by_node_id[node_id_text] = summary_text
        return summaries_by_node_id

    def anchors_for_node(self, node_id: str) -> dict[str, object] | None:
        """返回节点和它的源码锚点；节点不存在时返回 None。"""

        node = self.node_by_id(node_id)
        if node is None:
            return None
        source_file = self._source_file_for_node(node_id)
        return {
            "node": node,
            "source_file": source_file,
            "anchors": self._anchors_for_node(node_id, source_file=source_file),
        }

    def source_file(self, relative_path: str) -> dict[str, object] | None:
        rows = self._query(
            """
            MATCH (node:RigelNode:File)
            WHERE node.relative_path = $relative_path
            RETURN node.id, properties(node)
            LIMIT 1
            """,
            {"relative_path": relative_path},
        )
        if not rows:
            return None
        return format_source_file(format_node(rows[0][0], rows[0][1]))

    def node_by_id(self, node_id: str) -> dict[str, object] | None:
        rows = self._query(
            """
            MATCH (node:RigelNode)
            WHERE node.id = $node_id
            RETURN node.id, properties(node)
            LIMIT 1
            """,
            {"node_id": node_id},
        )
        if not rows:
            return None
        return format_node(rows[0][0], rows[0][1])

    def related_nodes(self, node_id: str, *, limit: int) -> list[dict[str, object]]:
        return self.expand_graph(
            node_id=node_id,
            direction="both",
            edge_types=list(VISIBLE_EDGE_TYPES),
            limit=limit,
        )["relations"]

    def neighbors(
        self,
        node_ids: list[str],
        *,
        direction: GraphExpansionDirection = "both",
        edge_types: list[str] | None = None,
        limit: int,
    ) -> dict[str, list[dict[str, object]]]:
        """按节点 ID 批量读取邻接关系。"""

        if not node_ids:
            return {}
        normalized_edge_types = _visible_edge_types(edge_types or list(VISIBLE_EDGE_TYPES))
        return {
            node_id: self.expand_graph(
                node_id=node_id,
                direction=direction,
                edge_types=normalized_edge_types,
                limit=limit,
            )["relations"]
            for node_id in dict.fromkeys(node_ids)
        }

    def paths(
        self,
        *,
        source_id: str,
        target_id: str,
        max_depth: int,
        limit: int,
    ) -> list[dict[str, object]]:
        """查找两个可视图节点之间的有向路径。"""

        if max_depth < 1:
            raise ValueError("max_depth 必须大于 0")
        rows = self._query(
            f"""
            MATCH path = (source:RigelNode)-[*1..{max_depth}]->(target:RigelNode)
            WHERE source.id = $source_id
              AND target.id = $target_id
              AND all(node IN nodes(path) WHERE node.rigel_type IN $visible_node_types)
              AND all(edge IN relationships(path) WHERE type(edge) IN $visible_edge_types)
            RETURN
              [node IN nodes(path) | [node.id, properties(node)]],
              [edge IN relationships(path) | [startNode(edge).id, endNode(edge).id, type(edge), properties(edge)]]
            LIMIT $limit
            """,
            {
                "source_id": source_id,
                "target_id": target_id,
                "visible_node_types": list(VISIBLE_NODE_TYPES),
                "visible_edge_types": list(VISIBLE_EDGE_TYPES),
                "limit": limit,
            },
        )
        return [
            {
                "nodes": [format_node(node_id, properties) for node_id, properties in node_rows],
                "edges": [
                    format_edge(edge_source_id, edge_target_id, edge_type, properties)
                    for edge_source_id, edge_target_id, edge_type, properties in edge_rows
                ],
            }
            for node_rows, edge_rows in rows
        ]

    def auto_complete(self, query: str, *, limit: int) -> list[dict[str, object]]:
        """按前缀搜索可视节点标签。"""

        query_text = query.strip().lower()
        if not query_text:
            return []
        rows = self._query(
            """
            MATCH (node:RigelNode)
            WHERE node.rigel_type IN $visible_node_types
              AND (
                toLower(coalesce(node.name, '')) STARTS WITH $query
                OR toLower(coalesce(node.display_name, '')) STARTS WITH $query
                OR toLower(coalesce(node.qualified_name, '')) STARTS WITH $query
                OR toLower(coalesce(node.relative_path, '')) STARTS WITH $query
              )
            RETURN node.id, properties(node)
            ORDER BY coalesce(node.display_name, node.qualified_name, node.relative_path, node.name, node.id)
            LIMIT $limit
            """,
            {
                "query": query_text,
                "visible_node_types": list(VISIBLE_NODE_TYPES),
                "limit": limit,
            },
        )
        return [format_node(node_id, properties) for node_id, properties in rows]

    def expand_graph(
        self,
        *,
        node_id: str,
        direction: GraphExpansionDirection,
        edge_types: list[str],
        limit: int,
    ) -> dict[str, object]:
        """按方向和边类型读取指定节点的一跳邻接关系。"""

        normalized_edge_types = _visible_edge_types(edge_types)
        rows = self._query(
            _expand_graph_query(direction),
            {
                "node_id": node_id,
                "visible_node_types": list(VISIBLE_NODE_TYPES),
                "edge_types": normalized_edge_types,
                "limit": limit,
            },
        )
        relations = [format_graph_relation(row, origin_node_id=node_id) for row in rows]
        return {"node_id": node_id, "relations": relations}

    def _anchors_for_node(
        self,
        node_id: str,
        *,
        source_file: dict[str, object] | None,
    ) -> list[dict[str, object]]:
        rows = self._query(
            """
            MATCH (owner:RigelNode)-[edge:HAS_ANCHOR]->(anchor:RigelNode:Anchor)
            WHERE owner.id = $node_id
            RETURN anchor.id, properties(anchor), properties(edge)
            """,
            {"node_id": node_id},
        )
        anchors = [
            format_anchor(anchor_id, anchor_properties, edge_properties, source_file)
            for anchor_id, anchor_properties, edge_properties in rows
        ]
        anchors.sort(key=anchor_sort_key)
        return anchors

    def _source_file_for_node(self, node_id: str) -> dict[str, object] | None:
        current_node_id: str | None = node_id
        visited_node_ids: set[str] = set()
        while current_node_id and current_node_id not in visited_node_ids:
            visited_node_ids.add(current_node_id)
            node = self.node_by_id(current_node_id)
            if node is None:
                return None
            if node["type"] == "File":
                return format_source_file(node)
            # 实体没有直接保存文件路径，沿 CONTAINS 父链回溯到最近的 File 节点。
            current_node_id = self._parent_node_id(current_node_id)
        return None

    def _parent_node_id(self, node_id: str) -> str | None:
        rows = self._query(
            """
            MATCH (parent:RigelNode)-[:CONTAINS]->(child:RigelNode)
            WHERE child.id = $node_id
            RETURN parent.id
            LIMIT 1
            """,
            {"node_id": node_id},
        )
        if not rows:
            return None
        return str(rows[0][0])

    def _scalar_query(self, query: str, parameters: Mapping[str, object]) -> int:
        rows = self._query(query, parameters)
        if not rows:
            return 0
        return int(rows[0][0])

    def _query(self, query: str, parameters: Mapping[str, object]) -> list[list[Any]]:
        result = self._store.graph.query(query, dict(parameters))
        return list(result.result_set)


def _visible_edge_types(edge_types: list[str]) -> list[str]:
    visible_edge_type_set = set(VISIBLE_EDGE_TYPES)
    return [
        edge_type
        for edge_type in dict.fromkeys(edge_types)
        if edge_type in visible_edge_type_set
    ]


def _with_generated_summary(node: dict[str, object], summaries_by_node_id: Mapping[str, str]) -> dict[str, object]:
    summary_text = summaries_by_node_id.get(str(node["id"]))
    if summary_text is None:
        return node

    formatted_node = dict(node)
    properties = dict(cast(Mapping[str, object], node["properties"]))
    properties["generated_summary"] = summary_text
    formatted_node["properties"] = properties
    return formatted_node


def _limit_rows(rows: list[list[Any]], limit: int | None) -> list[list[Any]]:
    if limit is None:
        return rows
    return rows[:limit]


def _visual_node_sort_key(row: list[Any]) -> tuple[int, int, str]:
    node_id, properties = row
    node_properties = cast(Mapping[str, object], properties)
    node_type = str(node_properties.get("rigel_type", ""))
    kind_norm = str(node_properties.get("kind_norm", ""))
    stable_label = str(
        node_properties.get("relative_path")
        or node_properties.get("qualified_name")
        or node_properties.get("display_name")
        or node_properties.get("name")
        or node_id
    )
    return (
        _visual_node_type_priority(node_type),
        _visual_entity_kind_priority(kind_norm),
        stable_label,
    )


def _visual_node_type_priority(node_type: str) -> int:
    priorities = {
        "Repository": 0,
        "Module": 1,
        "File": 2,
        "Entity": 3,
    }
    return priorities.get(node_type, len(priorities))


def _visual_entity_kind_priority(kind_norm: str) -> int:
    priorities = {
        "class": 0,
        "interface": 1,
        "enum": 2,
        "record": 3,
        "field": 4,
        "method": 5,
    }
    return priorities.get(kind_norm, len(priorities))


def _expand_graph_query(direction: GraphExpansionDirection) -> str:
    match direction:
        case "outgoing":
            node_filter = "source.id = $node_id"
        case "incoming":
            node_filter = "target.id = $node_id"
        case _:
            node_filter = "(source.id = $node_id OR target.id = $node_id)"

    return f"""
    MATCH (source:RigelNode)-[edge]->(target:RigelNode)
    WHERE {node_filter}
      AND source.rigel_type IN $visible_node_types
      AND target.rigel_type IN $visible_node_types
      AND type(edge) IN $edge_types
    RETURN source.id, properties(source), target.id, properties(target), type(edge), properties(edge)
    LIMIT $limit
    """
