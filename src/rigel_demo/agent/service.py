"""Agent 与 Web 共用的图谱证据读取服务。"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

from rigel_demo.indexing.retrieval_summaries import RETRIEVAL_SUMMARY_PURPOSE
from rigel_demo.storage.falkordb_store import FalkorDBConfig, FalkorDBStore

DEFAULT_GRAPH_LIMIT = 500
DEFAULT_RECALL_LIMIT = 8
DEFAULT_RECALL_EXPANSION_LIMIT = 3
DEFAULT_SOURCE_SLICE_MAX_LINES = 120
DEFAULT_CONTEXT_SOURCE_SLICE_LIMIT = 1
VISIBLE_NODE_TYPES = ("Repository", "Module", "File", "Entity")
VISIBLE_EDGE_TYPES = ("CONTAINS", "DEPENDS_ON", "SPECIALIZES", "ALIASES")

GraphExpansionDirection = Literal["incoming", "outgoing", "both"]


class SourcePathError(ValueError):
    """源码路径不在当前仓库内。"""


class SourceLineRangeError(ValueError):
    """源码切片行号范围不可用。"""


class SourceFileNotFoundError(FileNotFoundError):
    """源码文件不存在。"""


class SourceReadError(RuntimeError):
    """源码文件读取失败。"""


class RepositorySourceReader:
    """从当前仓库安全读取源码切片。"""

    def __init__(self, repository_path: Path) -> None:
        self._repository_path = repository_path.resolve()

    def normalize_relative_path(self, relative_path: str) -> str:
        candidate_path = self._resolve_repository_file(relative_path)
        try:
            return candidate_path.relative_to(self._repository_path).as_posix()
        except ValueError as error:
            raise SourcePathError("源码路径必须位于当前仓库内") from error

    def read_slice(
        self,
        relative_path: str,
        *,
        start_line: int,
        end_line: int,
        max_lines: int = DEFAULT_SOURCE_SLICE_MAX_LINES,
    ) -> dict[str, object]:
        normalized_path = self.normalize_relative_path(relative_path)
        _ensure_positive_int(max_lines, "max_lines")
        _ensure_positive_int(start_line, "start_line")
        _ensure_positive_int(end_line, "end_line")
        if end_line < start_line:
            raise SourceLineRangeError("end_line 必须大于或等于 start_line")

        bounded_end_line = min(end_line, start_line + max_lines - 1)
        file_path = self._repository_path / normalized_path
        if not file_path.exists() or not file_path.is_file():
            raise SourceFileNotFoundError(f"未找到源码文件：{normalized_path}")

        try:
            lines = file_path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError as error:
            raise SourceReadError(f"源码文件不是 UTF-8 文本：{normalized_path}") from error
        except OSError as error:
            raise SourceReadError(f"读取源码文件失败：{normalized_path}") from error

        total_lines = len(lines)
        if start_line > total_lines:
            raise SourceLineRangeError(f"start_line 超出文件总行数：{total_lines}")

        actual_end_line = min(bounded_end_line, total_lines)
        selected_lines = lines[start_line - 1 : actual_end_line]
        return {
            "relative_path": normalized_path,
            "start_line": start_line,
            "end_line": actual_end_line,
            "requested_end_line": end_line,
            "truncated": actual_end_line < end_line,
            "total_lines": total_lines,
            "content": "\n".join(selected_lines),
        }

    def _resolve_repository_file(self, relative_path: str) -> Path:
        if not relative_path.strip():
            raise SourcePathError("源码路径不能为空")
        raw_path = Path(relative_path)
        if raw_path.is_absolute():
            candidate_path = raw_path.resolve()
        else:
            candidate_path = (self._repository_path / raw_path).resolve()
        try:
            # resolve 后再 relative_to，可以同时拦截绝对路径和 `..` 逃逸。
            candidate_path.relative_to(self._repository_path)
        except ValueError as error:
            raise SourcePathError("源码路径必须位于当前仓库内") from error
        return candidate_path


def _ensure_positive_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SourceLineRangeError(f"{name} 必须大于 0")


class RigelGraphReader:
    """读取 FalkorDBLite 中的 Rigel 图谱数据。"""

    def __init__(self, *, database_path: Path, graph_name: str) -> None:
        self._database_path = database_path
        self._graph_name = graph_name

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

    def graph(self, *, limit: int) -> dict[str, list[dict[str, object]]]:
        """读取默认可视化语义节点和这些节点之间的语义边。"""

        node_rows = self._query(
            """
            MATCH (node:RigelNode)
            WHERE node.rigel_type IN $visible_node_types
            RETURN node.id, properties(node)
            LIMIT $limit
            """,
            {"visible_node_types": list(VISIBLE_NODE_TYPES), "limit": limit},
        )
        nodes = [_format_node(node_id, properties) for node_id, properties in node_rows]
        node_ids = [node["id"] for node in nodes]
        if not node_ids:
            return {"nodes": [], "edges": []}

        # 边只返回当前节点窗口内部的关系，避免前端收到指向缺失节点的悬空连线。
        edge_rows = self._query(
            """
            MATCH (source:RigelNode)-[edge]->(target:RigelNode)
            WHERE source.id IN $node_ids AND target.id IN $node_ids
              AND type(edge) IN $visible_edge_types
            RETURN source.id, target.id, type(edge), properties(edge)
            LIMIT $limit
            """,
            {
                "node_ids": node_ids,
                "visible_edge_types": list(VISIBLE_EDGE_TYPES),
                "limit": limit * 2,
            },
        )
        edges = [_format_edge(source_id, target_id, edge_type, properties) for source_id, target_id, edge_type, properties in edge_rows]
        return {"nodes": nodes, "edges": edges}

    def recall(
        self,
        query_embedding: list[float],
        *,
        embedding_model: str,
        limit: int,
        expansion_limit: int,
    ) -> list[dict[str, object]]:
        """通过 Summary embedding 召回种子节点并补充一跳图谱上下文。"""

        rows = self._query(
            """
            CALL db.idx.vector.queryNodes('Summary', 'embedding', $vector_limit, vecf32($query_embedding))
            YIELD node AS summary, score AS distance
            MATCH (summary)-[:DESCRIBES]->(target:RigelNode)
            WHERE summary.purpose = $purpose
              AND summary.embedding_model = $embedding_model
              AND summary.embedding_dimensions = $embedding_dimensions
              AND target.rigel_type IN $visible_node_types
            RETURN summary.id, properties(summary), target.id, properties(target), distance
            ORDER BY distance ASC
            """,
            {
                "purpose": RETRIEVAL_SUMMARY_PURPOSE,
                "embedding_model": embedding_model,
                "embedding_dimensions": len(query_embedding),
                "visible_node_types": list(VISIBLE_NODE_TYPES),
                "query_embedding": query_embedding,
                "vector_limit": limit,
            },
        )
        scored_results: list[dict[str, object]] = []
        for summary_id, summary_properties, target_id, target_properties, distance in rows:
            score = _cosine_distance_to_similarity(distance)
            if score <= 0:
                continue
            node = _format_node(target_id, target_properties)
            scored_results.append(
                {
                    "score": score,
                    "summary": _format_summary(summary_id, summary_properties),
                    "node": node,
                    "related": self.related_nodes(target_id, limit=expansion_limit),
                }
            )

        scored_results.sort(
            key=lambda result: (
                -cast(float, result["score"]),
                str(cast(Mapping[str, object], result["node"])["label"]),
            )
        )
        return scored_results[:limit]

    def context(
        self,
        *,
        query: str,
        query_embedding: list[float],
        embedding_model: str,
        limit: int,
        expansion_limit: int,
        source_reader: RepositorySourceReader,
    ) -> dict[str, object]:
        """组装 Agent 可直接引用的结构化图谱上下文。"""

        # Agent 工具接口不生成自然语言答案，只返回可追溯的图谱证据和源码切片。
        recall_results = self.recall(
            query_embedding,
            embedding_model=embedding_model,
            limit=limit,
            expansion_limit=expansion_limit,
        )
        return {
            "query": query,
            "strategy": "vector_recall",
            "seeds": [
                self._context_seed_from_recall_result(index, result, source_reader=source_reader)
                for index, result in enumerate(recall_results, start=1)
            ],
        }

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
        return _format_source_file(_format_node(rows[0][0], rows[0][1]))

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
        return _format_node(rows[0][0], rows[0][1])

    def related_nodes(self, node_id: str, *, limit: int) -> list[dict[str, object]]:
        return self.expand_graph(
            node_id=node_id,
            direction="both",
            edge_types=list(VISIBLE_EDGE_TYPES),
            limit=limit,
        )["relations"]

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
        if direction == "outgoing":
            query = """
            MATCH (source:RigelNode)-[edge]->(target:RigelNode)
            WHERE source.id = $node_id
              AND source.rigel_type IN $visible_node_types
              AND target.rigel_type IN $visible_node_types
              AND type(edge) IN $edge_types
            RETURN source.id, properties(source), target.id, properties(target), type(edge), properties(edge)
            LIMIT $limit
            """
        elif direction == "incoming":
            query = """
            MATCH (source:RigelNode)-[edge]->(target:RigelNode)
            WHERE target.id = $node_id
              AND source.rigel_type IN $visible_node_types
              AND target.rigel_type IN $visible_node_types
              AND type(edge) IN $edge_types
            RETURN source.id, properties(source), target.id, properties(target), type(edge), properties(edge)
            LIMIT $limit
            """
        else:
            query = """
            MATCH (source:RigelNode)-[edge]->(target:RigelNode)
            WHERE (source.id = $node_id OR target.id = $node_id)
              AND source.rigel_type IN $visible_node_types
              AND target.rigel_type IN $visible_node_types
              AND type(edge) IN $edge_types
            RETURN source.id, properties(source), target.id, properties(target), type(edge), properties(edge)
            LIMIT $limit
            """

        rows = self._query(
            query,
            {
                "node_id": node_id,
                "visible_node_types": list(VISIBLE_NODE_TYPES),
                "edge_types": normalized_edge_types,
                "limit": limit,
            },
        )
        relations: list[dict[str, object]] = []
        for source_id, source_properties, target_id, target_properties, edge_type, edge_properties in rows:
            source_node = _format_node(source_id, source_properties)
            target_node = _format_node(target_id, target_properties)
            related_node = target_node if source_id == node_id else source_node
            relations.append(
                {
                    "direction": "outgoing" if source_id == node_id else "incoming",
                    "edge": _format_edge(source_id, target_id, edge_type, edge_properties),
                    "node": related_node,
                }
            )
        return {"node_id": node_id, "relations": relations}

    def _context_seed_from_recall_result(
        self,
        rank: int,
        result: dict[str, object],
        *,
        source_reader: RepositorySourceReader,
    ) -> dict[str, object]:
        node = cast(dict[str, object], result["node"])
        node_id = str(node["id"])
        source_file = self._source_file_for_node(node_id)
        anchors = self._anchors_for_node(node_id, source_file=source_file)
        return {
            "rank": rank,
            "score": result["score"],
            "summary": result["summary"],
            "node": node,
            "related": result["related"],
            "source_file": source_file,
            "anchors": anchors,
            "source_slices": _source_slices_for_anchors(anchors, source_reader=source_reader),
        }

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
            _format_anchor(anchor_id, anchor_properties, edge_properties, source_file)
            for anchor_id, anchor_properties, edge_properties in rows
        ]
        anchors.sort(key=_anchor_sort_key)
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
                return _format_source_file(node)
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
        store = FalkorDBStore.connect(
            FalkorDBConfig(graph_name=self._graph_name, database_path=str(self._database_path))
        )
        result = store.graph.query(query, dict(parameters))
        return list(result.result_set)


def _source_slices_for_anchors(
    anchors: list[dict[str, object]],
    *,
    source_reader: RepositorySourceReader,
) -> list[dict[str, object]]:
    source_slices: list[dict[str, object]] = []
    for anchor in anchors[:DEFAULT_CONTEXT_SOURCE_SLICE_LIMIT]:
        source_file = cast(dict[str, object], anchor["source_file"])
        relative_path = cast(str, source_file["relative_path"])
        try:
            source_slice = source_reader.read_slice(
                relative_path,
                start_line=int(anchor["start_line"]),
                end_line=int(anchor["end_line"]),
            )
        except (SourcePathError, SourceLineRangeError, SourceFileNotFoundError, SourceReadError):
            continue
        source_slices.append(
            {
                **source_slice,
                "anchor": {
                    "id": anchor["id"],
                    "role": anchor["role"],
                    "start_line": anchor["start_line"],
                    "start_col": anchor["start_col"],
                    "end_line": anchor["end_line"],
                    "end_col": anchor["end_col"],
                },
                "source_file": source_file,
            }
        )
    return source_slices


def _visible_edge_types(edge_types: list[str]) -> list[str]:
    return edge_types


def _read_property(properties: Mapping[str, object], name: str) -> str:
    return cast(str, properties[name])


def _format_node(node_id: str, properties: Mapping[str, object]) -> dict[str, object]:
    formatted_properties = dict(properties)
    return {
        "id": node_id,
        "type": str(formatted_properties["rigel_type"]),
        "label": _node_label(node_id, formatted_properties),
        "properties": formatted_properties,
    }


def _format_edge(
    source_id: str,
    target_id: str,
    edge_type: str,
    properties: Mapping[str, object],
) -> dict[str, object]:
    formatted_properties = dict(properties)
    return {
        "id": str(formatted_properties["id"]),
        "source": source_id,
        "target": target_id,
        "type": edge_type,
        "properties": formatted_properties,
    }


def _format_summary(summary_id: str, properties: Mapping[str, object]) -> dict[str, object]:
    return {
        "id": summary_id,
        "text": _read_property(properties, "text"),
        "summary_model": _read_property(properties, "summary_model"),
        "embedding_model": _read_property(properties, "embedding_model"),
        "embedding_dimensions": _read_int_property(properties, "embedding_dimensions"),
        "source_hash": _read_property(properties, "source_hash"),
    }


def _format_anchor(
    anchor_id: str,
    anchor_properties: Mapping[str, object],
    edge_properties: Mapping[str, object],
    source_file: dict[str, object] | None,
) -> dict[str, object]:
    return {
        "id": anchor_id,
        "role": _read_property(edge_properties, "role"),
        "start_line": _read_int_property(anchor_properties, "start_line"),
        "start_col": _read_int_property(anchor_properties, "start_col"),
        "end_line": _read_int_property(anchor_properties, "end_line"),
        "end_col": _read_int_property(anchor_properties, "end_col"),
        "source_file": source_file,
        "properties": dict(anchor_properties),
    }


def _format_source_file(file_node: Mapping[str, object]) -> dict[str, object]:
    properties = cast(Mapping[str, object], file_node["properties"])
    return {
        "id": file_node["id"],
        "label": file_node["label"],
        "relative_path": _read_property(properties, "relative_path"),
        "language": _read_property(properties, "language"),
        "content_hash": _read_property(properties, "content_hash"),
        "position_encoding": _read_property(properties, "position_encoding"),
    }


def _anchor_sort_key(anchor: Mapping[str, object]) -> tuple[int, int, str]:
    return (
        _anchor_role_priority(str(anchor["role"])),
        int(anchor["start_line"]),
        str(anchor["id"]),
    )


def _anchor_role_priority(role: str) -> int:
    priorities = {
        "definition": 0,
        "body": 1,
        "name": 2,
    }
    return priorities[role]


def _read_int_property(properties: Mapping[str, object], name: str) -> int:
    return cast(int, properties[name])


def _cosine_distance_to_similarity(distance: float) -> float:
    return max(0.0, 1.0 - float(distance))


def _node_label(node_id: str, properties: Mapping[str, object]) -> str:
    node_type = _read_property(properties, "rigel_type")
    label_properties = {
        "Repository": "name",
        "Module": "name",
        "File": "relative_path",
        "Entity": "display_name",
        "Anchor": "role",
        "Summary": "text",
    }
    return _read_property(properties, label_properties[node_type])
