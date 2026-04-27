"""FalkorDB 图谱写入"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rigel_demo.core.graph_ir import GraphEdge, GraphIR, GraphNode, JsonObject, JsonValue


RIGEL_NODE_LABEL = "RigelNode"


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

        # FalkorDB 写边前必须能 MATCH 到两端节点，因此这里固定先写节点再写关系。
        for node in graph_ir.nodes:
            self.upsert_node(node)
        for edge in graph_ir.edges:
            self.upsert_edge(edge)

    def upsert_node(self, node: GraphNode) -> None:
        """写入或更新单个 GraphIR 节点。"""

        node_type = node.type.value
        properties = _database_properties(
            {
                "id": node.id,
                "rigel_type": node.type.value,
                **node.properties,
            }
        )
        # node_type 来自 GraphIR 枚举，不接受外部输入；属性值统一走参数化绑定。
        self._graph.query(
            f"""
            MERGE (node:{RIGEL_NODE_LABEL}:{node_type} {{id: $id}})
            SET node += $properties
            """,
            {"id": node.id, "properties": properties},
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


def _database_properties(properties: JsonObject) -> dict[str, object]:
    """把 GraphIR 属性转换成 FalkorDB 可稳定保存的属性。"""

    return {key: _database_value(value) for key, value in properties.items()}


def _database_value(value: JsonValue) -> object:
    """保留标量属性，把复合属性序列化为 JSON 字符串。"""

    if isinstance(value, str | int | float | bool) or value is None:
        return value
    # FalkorDB 属性以标量为主，复合值转成排序后的 JSON 字符串，确保重复写入结果稳定。
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
