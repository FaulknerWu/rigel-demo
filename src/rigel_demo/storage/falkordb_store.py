"""FalkorDB 图谱写入"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rigel_demo.core.graph_ir import GraphEdge, GraphIR, GraphNode, JsonObject, JsonValue


RIGEL_NODE_LABEL = "RigelNode"
SUMMARY_NODE_LABEL = "Summary"
SUMMARY_EMBEDDING_PROPERTY = "embedding"
SUMMARY_VECTOR_SIMILARITY_FUNCTION = "cosine"


@dataclass(frozen=True, slots=True)
class FalkorDBConfig:
    """FalkorDBLite 连接配置。"""

    graph_name: str = "rigel"
    database_path: str = ".rigel/falkordb.db"


class FalkorDBStore:
    """把 GraphIR 幂等写入 FalkorDB。"""

    def __init__(self, graph: Any) -> None:
        self._graph = graph

    @property
    def graph(self) -> Any:
        """暴露底层 FalkorDB 图对象，供只读查询复用。"""

        return self._graph

    @classmethod
    def connect(cls, config: FalkorDBConfig) -> "FalkorDBStore":
        """使用 FalkorDBLite 创建本地图谱写入器。"""

        from redislite.falkordb_client import FalkorDB

        database_path = Path(config.database_path)
        database_path.parent.mkdir(parents=True, exist_ok=True)
        client = FalkorDB(str(database_path))
        return cls(client.select_graph(config.graph_name))

    def upsert_graph(self, graph_ir: GraphIR) -> None:
        """按节点再边的顺序幂等写入完整 GraphIR。"""

        summary_embedding_dimensions = _graph_summary_embedding_dimensions(graph_ir)
        # FalkorDB 写边前必须能 MATCH 到两端节点，因此这里固定先写节点再写关系。
        for node in graph_ir.nodes:
            self.upsert_node(node)
        for edge in graph_ir.edges:
            self.upsert_edge(edge)
        if summary_embedding_dimensions is not None:
            self.ensure_summary_vector_index(dimensions=summary_embedding_dimensions)

    def upsert_node(self, node: GraphNode) -> None:
        """写入或更新单个 GraphIR 节点。"""

        node_type = node.type.value
        raw_properties: JsonObject = {
            "id": node.id,
            "rigel_type": node.type.value,
            **node.properties,
        }
        native_embedding = _summary_embedding(raw_properties) if node_type == SUMMARY_NODE_LABEL else None
        if native_embedding is not None:
            raw_properties = dict(raw_properties)
            raw_properties.pop(SUMMARY_EMBEDDING_PROPERTY, None)

        properties = _database_properties(raw_properties)
        # node_type 来自 GraphIR 枚举，不接受外部输入；属性值统一走参数化绑定。
        self._graph.query(
            f"""
            MERGE (node:{RIGEL_NODE_LABEL}:{node_type} {{id: $id}})
            SET node += $properties
            """,
            {"id": node.id, "properties": properties},
        )
        if native_embedding is not None:
            self._graph.query(
                f"""
                MATCH (node:{RIGEL_NODE_LABEL}:{node_type} {{id: $id}})
                SET node.{SUMMARY_EMBEDDING_PROPERTY} = vecf32($embedding)
                """,
                {"id": node.id, "embedding": native_embedding},
            )

    def upsert_edge(self, edge: GraphEdge) -> None:
        """写入或更新单条 GraphIR 边。"""

        edge_type = edge.type.value
        properties = _database_properties(
            {
                "id": edge.id,
                "rigel_type": edge.type.value,
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                **edge.properties,
            }
        )
        # edge_type 来自 GraphIR 枚举，不接受外部输入；属性值统一走参数化绑定。
        self._graph.query(
            f"""
            MATCH (source:{RIGEL_NODE_LABEL} {{id: $source_id}})
            MATCH (target:{RIGEL_NODE_LABEL} {{id: $target_id}})
            MERGE (source)-[edge:{edge_type} {{id: $id}}]->(target)
            SET edge += $properties
            """,
            {
                "id": edge.id,
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "properties": properties,
            },
        )

    def ensure_summary_vector_index(self, *, dimensions: int) -> None:
        """确保 Summary.embedding 上存在 FalkorDB 原生向量索引。"""

        if dimensions <= 0:
            raise ValueError("Summary 向量索引维度必须大于 0")

        if self._summary_vector_index_exists(dimensions=dimensions):
            return

        self._graph.query(
            f"""
            CREATE VECTOR INDEX FOR (summary:{SUMMARY_NODE_LABEL})
            ON (summary.{SUMMARY_EMBEDDING_PROPERTY})
            OPTIONS {{
                dimension: $dimensions,
                similarityFunction: $similarity_function
            }}
            """,
            {
                "dimensions": dimensions,
                "similarity_function": SUMMARY_VECTOR_SIMILARITY_FUNCTION,
            },
        )

    def _summary_vector_index_exists(self, *, dimensions: int) -> bool:
        rows = self._graph.query("CALL db.indexes()").result_set
        for row in rows:
            if len(row) < 4:
                continue
            label, _properties, index_types, options = row[:4]
            if label != SUMMARY_NODE_LABEL:
                continue
            if not _embedding_has_vector_index(index_types):
                continue
            embedding_options = _embedding_index_options(options)
            if not embedding_options:
                return True
            indexed_dimensions = embedding_options.get("dimension")
            similarity_function = embedding_options.get("similarityFunction")
            if indexed_dimensions == dimensions and similarity_function == SUMMARY_VECTOR_SIMILARITY_FUNCTION:
                return True
        return False


def _database_properties(properties: JsonObject) -> dict[str, object]:
    """把 GraphIR 属性转换成 FalkorDB 可稳定保存的属性。"""

    return {key: _database_value(value) for key, value in properties.items()}


def _database_value(value: JsonValue) -> object:
    """保留标量属性，把复合属性序列化为 JSON 字符串。"""

    if isinstance(value, str | int | float | bool) or value is None:
        return value
    # FalkorDB 属性以标量为主，复合值转成排序后的 JSON 字符串，确保重复写入结果稳定。
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _summary_embedding(properties: JsonObject) -> list[float] | None:
    value = properties.get(SUMMARY_EMBEDDING_PROPERTY)
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("Summary.embedding 必须是向量列表")

    embedding: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise ValueError("Summary.embedding 只能包含数字")
        embedding.append(float(item))
    if not embedding:
        raise ValueError("Summary.embedding 不能为空")
    return embedding


def _graph_summary_embedding_dimensions(graph_ir: GraphIR) -> int | None:
    dimensions: set[int] = set()
    for node in graph_ir.nodes:
        if node.type.value != SUMMARY_NODE_LABEL:
            continue
        embedding = _summary_embedding(node.properties)
        if embedding is not None:
            dimensions.add(len(embedding))

    if not dimensions:
        return None
    if len(dimensions) > 1:
        raise ValueError("同一图谱中的 Summary.embedding 维度必须一致")
    return next(iter(dimensions))


def _embedding_has_vector_index(index_types: object) -> bool:
    if not isinstance(index_types, dict):
        return False
    property_index_types = index_types.get(SUMMARY_EMBEDDING_PROPERTY)
    return isinstance(property_index_types, list) and "VECTOR" in property_index_types


def _embedding_index_options(options: object) -> dict[str, object]:
    if not isinstance(options, dict):
        return {}
    embedding_options = options.get(SUMMARY_EMBEDDING_PROPERTY)
    return dict(embedding_options) if isinstance(embedding_options, dict) else {}
