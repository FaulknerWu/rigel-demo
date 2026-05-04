"""图查询结果的 API 展示格式。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast


def format_graph_relation(row: list[Any], *, origin_node_id: str) -> dict[str, object]:
    source_id, source_properties, target_id, target_properties, edge_type, edge_properties = row
    source_node = format_node(source_id, source_properties)
    target_node = format_node(target_id, target_properties)
    is_outgoing = source_id == origin_node_id
    return {
        "direction": "outgoing" if is_outgoing else "incoming",
        "edge": format_edge(source_id, target_id, edge_type, edge_properties),
        "node": target_node if is_outgoing else source_node,
    }


def format_node(node_id: str, properties: Mapping[str, object]) -> dict[str, object]:
    formatted_properties = dict(properties)
    return {
        "id": node_id,
        "type": str(formatted_properties["rigel_type"]),
        "label": node_label(node_id, formatted_properties),
        "properties": formatted_properties,
    }


def format_edge(
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


def format_summary(summary_id: str, properties: Mapping[str, object]) -> dict[str, object]:
    return {
        "id": summary_id,
        "text": read_property(properties, "text"),
        "summary_model": read_property(properties, "summary_model"),
        "embedding_model": read_property(properties, "embedding_model"),
        "embedding_dimensions": read_int_property(properties, "embedding_dimensions"),
        "source_hash": read_property(properties, "source_hash"),
    }


def format_anchor(
    anchor_id: str,
    anchor_properties: Mapping[str, object],
    edge_properties: Mapping[str, object],
    source_file: dict[str, object] | None,
) -> dict[str, object]:
    return {
        "id": anchor_id,
        "role": read_property(edge_properties, "role"),
        "start_line": read_int_property(anchor_properties, "start_line"),
        "start_col": read_int_property(anchor_properties, "start_col"),
        "end_line": read_int_property(anchor_properties, "end_line"),
        "end_col": read_int_property(anchor_properties, "end_col"),
        "source_file": source_file,
        "properties": dict(anchor_properties),
    }


def format_source_file(file_node: Mapping[str, object]) -> dict[str, object]:
    properties = cast(Mapping[str, object], file_node["properties"])
    return {
        "id": file_node["id"],
        "label": file_node["label"],
        "relative_path": read_property(properties, "relative_path"),
        "language": read_property(properties, "language"),
        "content_hash": read_property(properties, "content_hash"),
        "position_encoding": read_property(properties, "position_encoding"),
    }


def anchor_sort_key(anchor: Mapping[str, object]) -> tuple[int, int, str]:
    return (
        anchor_role_priority(str(anchor["role"])),
        int(anchor["start_line"]),
        str(anchor["id"]),
    )


def cosine_distance_to_similarity(distance: float) -> float:
    return max(0.0, 1.0 - float(distance))


def read_property(properties: Mapping[str, object], name: str) -> str:
    return cast(str, properties[name])


def read_int_property(properties: Mapping[str, object], name: str) -> int:
    return cast(int, properties[name])


def anchor_role_priority(role: str) -> int:
    priorities = {
        "definition": 0,
        "body": 1,
        "name": 2,
    }
    return priorities.get(role, len(priorities))


def node_label(node_id: str, properties: Mapping[str, object]) -> str:
    node_type = read_property(properties, "rigel_type")
    label_properties = {
        "Repository": "name",
        "Module": "name",
        "File": "relative_path",
        "Entity": "display_name",
        "Anchor": "role",
        "Summary": "text",
    }
    return read_property(properties, label_properties[node_type])
