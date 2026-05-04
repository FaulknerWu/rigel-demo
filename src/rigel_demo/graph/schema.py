"""Rigel 图谱 schema 映射。"""

from __future__ import annotations

from dataclasses import dataclass

from rigel_demo.graph.ir import EdgeType, NodeType


RIGEL_NODE_LABEL = "RigelNode"
SUMMARY_NODE_LABEL = NodeType.SUMMARY.value
SUMMARY_EMBEDDING_PROPERTY = "embedding"
SUMMARY_VECTOR_SIMILARITY_FUNCTION = "cosine"


@dataclass(frozen=True, slots=True)
class FalkorNodeSchema:
    """GraphIR 节点到 FalkorDB label 的映射。"""

    common_label: str
    type_label: str


def node_schema(node_type: NodeType) -> FalkorNodeSchema:
    return FalkorNodeSchema(common_label=RIGEL_NODE_LABEL, type_label=node_type.value)


def relationship_type(edge_type: EdgeType) -> str:
    return edge_type.value
