"""GraphRAG chat 的工具定义、参数校验与证据格式化。"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

from rigel_demo.graphrag.models import GraphRAGTrace
from rigel_demo.graphrag.state import GraphRAGState, ToolCall
from rigel_demo.llm import LLMMessage
from rigel_demo.prompts import (
    EXPAND_NEIGHBORS_TOOL_DESCRIPTION,
    QUERY_RELATION_TOOL_DESCRIPTION,
    RIGEL_EVIDENCE_CONTINUATION_PROMPT,
    RIGEL_TOOL_SYSTEM_INSTRUCTION,
    VECTOR_SEARCH_SEEDS_TOOL_DESCRIPTION,
)
from rigel_demo.project.summaries import RETRIEVAL_SUMMARY_PURPOSE
from rigel_demo.query.presentation import (
    format_edge,
    format_node,
    format_summary,
)
from rigel_demo.query.service import GraphExpansionDirection, VISIBLE_EDGE_TYPES, VISIBLE_NODE_TYPES

RIGEL_ALLOWED_NODE_TYPES = ("Repository", "Module", "File", "Entity", "Anchor", "Summary")
RIGEL_ALLOWED_RELATIONSHIPS = ("CONTAINS", "DEPENDS_ON", "SPECIALIZES", "ALIASES", "HAS_ANCHOR", "DESCRIBES")
MAX_TOTAL_TOOL_CALLS = 6
TOOL_CALL_LIMITS = {
    "vector_search_seeds": 1,
    "expand_neighbors": 3,
    "query_relation": 3,
}
TOOL_EDGE_TYPES = tuple(VISIBLE_EDGE_TYPES)
TOOL_NAMES = tuple(TOOL_CALL_LIMITS)


def rigel_schema_instruction(schema_text: str) -> str:
    return (
        f"{schema_text}\n\n"
        f"Rigel 允许的节点类型：{', '.join(RIGEL_ALLOWED_NODE_TYPES)}。\n"
        f"Rigel 允许的关系类型：{', '.join(RIGEL_ALLOWED_RELATIONSHIPS)}。"
    )


def bind_tool_schemas(chat_model: Any) -> Any:
    bind_tools = getattr(chat_model, "bind_tools", None)
    if not callable(bind_tools):
        return chat_model
    return bind_tools(rigel_tools())


def rigel_tools() -> list[Any]:
    from langchain_core.tools import tool

    @tool(description=VECTOR_SEARCH_SEEDS_TOOL_DESCRIPTION)
    def vector_search_seeds(query_texts: list[str]) -> str:
        return ""

    @tool(description=EXPAND_NEIGHBORS_TOOL_DESCRIPTION)
    def expand_neighbors(
        node_ids: list[str],
        direction: Literal["incoming", "outgoing", "both"] = "both",
        edge_types: list[str] | None = None,
    ) -> str:
        return ""

    @tool(description=QUERY_RELATION_TOOL_DESCRIPTION)
    def query_relation(
        node_ids: list[str],
        relation_type: str,
        direction: Literal["incoming", "outgoing", "both"] = "both",
    ) -> str:
        return ""

    return [vector_search_seeds, expand_neighbors, query_relation]


def initial_agent_messages(
    *,
    messages: Sequence[LLMMessage],
    schema: str,
    system_prompt: str,
) -> list[Any]:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    agent_messages: list[Any] = [
        SystemMessage(content=f"{system_prompt}\n\n{RIGEL_TOOL_SYSTEM_INSTRUCTION.format(ontology=schema)}")
    ]
    for message in messages:
        if message.role == "user":
            agent_messages.append(HumanMessage(content=message.content))
        else:
            agent_messages.append(AIMessage(content=message.content))
    return agent_messages


def agent_messages_with_evidence(state: GraphRAGState) -> list[Any]:
    from langchain_core.messages import SystemMessage

    evidence_message = SystemMessage(
        content=f"{RIGEL_EVIDENCE_CONTINUATION_PROMPT}\n{json_dumps(evidence_prompt_payload(state))}"
    )
    return [*state["agent_messages"], evidence_message]


def evidence_prompt_payload(state: GraphRAGState) -> dict[str, object]:
    return {
        "evidence": state.get("evidence", []),
        "known_node_ids": state.get("known_node_ids", []),
        "visited_node_ids": state.get("visited_node_ids", []),
        "tool_call_count": state.get("tool_call_count", 0),
        "limit_reached": state.get("limit_reached", False),
    }


def extract_tool_calls(response: object) -> list[ToolCall]:
    raw_tool_calls = getattr(response, "tool_calls", None)
    if isinstance(raw_tool_calls, list):
        return [normalize_tool_call(index, raw_tool_call) for index, raw_tool_call in enumerate(raw_tool_calls)]

    additional_kwargs = getattr(response, "additional_kwargs", None)
    if isinstance(additional_kwargs, Mapping) and isinstance(additional_kwargs.get("tool_calls"), list):
        return [
            normalize_openai_tool_call(index, raw_tool_call)
            for index, raw_tool_call in enumerate(cast(list[object], additional_kwargs["tool_calls"]))
        ]
    return []


def normalize_tool_call(index: int, raw_tool_call: object) -> ToolCall:
    if isinstance(raw_tool_call, Mapping):
        name = str(raw_tool_call.get("name", ""))
        args = raw_tool_call.get("args", {})
        tool_call_id = str(raw_tool_call.get("id") or f"tool_call:{index}")
        return ToolCall(id=tool_call_id, name=name, args=dict(args) if isinstance(args, Mapping) else {})
    name = str(getattr(raw_tool_call, "name", ""))
    args = getattr(raw_tool_call, "args", {})
    tool_call_id = str(getattr(raw_tool_call, "id", "") or f"tool_call:{index}")
    return ToolCall(id=tool_call_id, name=name, args=dict(args) if isinstance(args, Mapping) else {})


def normalize_openai_tool_call(index: int, raw_tool_call: object) -> ToolCall:
    if not isinstance(raw_tool_call, Mapping):
        return ToolCall(id=f"tool_call:{index}", name="", args={})
    function = raw_tool_call.get("function")
    if not isinstance(function, Mapping):
        return ToolCall(id=str(raw_tool_call.get("id") or f"tool_call:{index}"), name="", args={})
    raw_args = function.get("arguments")
    args = json.loads(raw_args) if isinstance(raw_args, str) and raw_args.strip() else {}
    return ToolCall(
        id=str(raw_tool_call.get("id") or f"tool_call:{index}"),
        name=str(function.get("name", "")),
        args=args if isinstance(args, dict) else {},
    )


def tool_message(result: Mapping[str, object], tool_call_id: str) -> Any:
    from langchain_core.messages import ToolMessage

    return ToolMessage(content=json_dumps(result), tool_call_id=tool_call_id)


def tool_result(tool_name: str, items: list[dict[str, object]], warnings: list[str]) -> dict[str, object]:
    return {
        "tool": tool_name,
        "items": items,
        "warnings": warnings,
    }


def trace_from_tool_result(
    tool_name: str,
    args: Mapping[str, object],
    result: Mapping[str, object],
) -> GraphRAGTrace:
    items = cast(list[dict[str, object]], result["items"])
    warnings = cast(list[str], result["warnings"])
    match tool_name:
        case "vector_search_seeds":
            trace_args: dict[str, object] = {
                "query_texts": string_list_arg(args, "query_texts"),
                "items": items,
            }
        case "expand_neighbors":
            trace_args = {
                "node_ids": string_list_arg(args, "node_ids"),
                "direction": direction_arg(args),
                "edge_types": edge_types_arg(args.get("edge_types"), default=list(TOOL_EDGE_TYPES))[0],
                "relations": items,
            }
        case "query_relation":
            trace_args = {
                "node_ids": string_list_arg(args, "node_ids"),
                "relation_type": string_arg(args, "relation_type", default=""),
                "direction": direction_arg(args),
                "relations": items,
            }
        case _:
            trace_args = {"items": items}
    if warnings:
        trace_args["warnings"] = warnings
    return GraphRAGTrace(name=tool_name, args=trace_args)


def string_arg(args: Mapping[str, object], name: str, *, default: str) -> str:
    value = args.get(name)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def string_list_arg(args: Mapping[str, object], name: str) -> list[str]:
    value = args.get(name)
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def direction_arg(args: Mapping[str, object]) -> GraphExpansionDirection:
    value = args.get("direction")
    if value in {"incoming", "outgoing", "both"}:
        return cast(GraphExpansionDirection, value)
    return "both"


def edge_types_arg(value: object, *, default: list[str]) -> tuple[list[str], list[str]]:
    if value is None:
        return default, []
    requested_edge_types = value
    if not isinstance(requested_edge_types, list):
        return default, ["edge_types 格式非法，已使用默认边类型"]
    normalized_edge_types = [
        edge_type.strip()
        for edge_type in requested_edge_types
        if isinstance(edge_type, str) and edge_type.strip() in TOOL_EDGE_TYPES
    ]
    ignored_edge_types = [
        str(edge_type)
        for edge_type in requested_edge_types
        if not isinstance(edge_type, str) or edge_type.strip() not in TOOL_EDGE_TYPES
    ]
    warnings = [f"非法边类型已忽略：{edge_type}" for edge_type in ignored_edge_types]
    return list(dict.fromkeys(normalized_edge_types)), warnings


def filter_known_node_ids(
    node_ids: list[str],
    *,
    known_node_ids: set[str],
    warnings: list[str],
) -> list[str]:
    accepted_node_ids: list[str] = []
    for node_id in dict.fromkeys(node_ids):
        if node_id not in known_node_ids:
            warnings.append(f"未知节点已忽略：{node_id}")
            continue
        accepted_node_ids.append(node_id)
    return accepted_node_ids


def filter_known_unvisited_node_ids(
    node_ids: list[str],
    *,
    known_node_ids: set[str],
    visited_node_ids: set[str],
    warnings: list[str],
) -> list[str]:
    accepted_node_ids: list[str] = []
    for node_id in filter_known_node_ids(node_ids, known_node_ids=known_node_ids, warnings=warnings):
        if node_id in visited_node_ids:
            warnings.append(f"已扩展节点已忽略：{node_id}")
            continue
        accepted_node_ids.append(node_id)
    return accepted_node_ids


def query_graph(graph: Any, query: str, params: Mapping[str, object]) -> list[dict[str, object] | list[object]]:
    result = graph.query(query, dict(params))
    if isinstance(result, list):
        return result
    result_set = getattr(result, "result_set", None)
    if isinstance(result_set, list):
        return result_set
    data = getattr(result, "data", None)
    if isinstance(data, list):
        return data
    return []


def seed_row_values(row: dict[str, object] | list[object]) -> tuple[object, object, object, object, object]:
    if isinstance(row, Mapping):
        return (
            row["summary_id"],
            row["summary_properties"],
            row["node_id"],
            row["node_properties"],
            row["distance"],
        )
    return row[0], row[1], row[2], row[3], row[4]


def relation_from_row(row: dict[str, object] | list[object], *, origin_node_id: str) -> dict[str, object]:
    if isinstance(row, Mapping):
        source_id = str(row["source_id"])
        source_properties = cast(Mapping[str, object], row["source_properties"])
        target_id = str(row["target_id"])
        target_properties = cast(Mapping[str, object], row["target_properties"])
        edge_type = str(row["edge_type"])
        edge_properties = cast(Mapping[str, object], row["edge_properties"])
    else:
        source_id = str(row[0])
        source_properties = cast(Mapping[str, object], row[1])
        target_id = str(row[2])
        target_properties = cast(Mapping[str, object], row[3])
        edge_type = str(row[4])
        edge_properties = cast(Mapping[str, object], row[5])

    source_node = format_node(source_id, source_properties)
    target_node = format_node(target_id, target_properties)
    is_outgoing = source_id == origin_node_id
    return {
        "tool": "graph_relation",
        "origin_node_id": origin_node_id,
        "direction": "outgoing" if is_outgoing else "incoming",
        "edge": format_edge(source_id, target_id, edge_type, edge_properties),
        "node": target_node if is_outgoing else source_node,
    }


def expand_graph_query(direction: GraphExpansionDirection) -> str:
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
    RETURN source.id AS source_id, properties(source) AS source_properties,
           target.id AS target_id, properties(target) AS target_properties,
           type(edge) AS edge_type, properties(edge) AS edge_properties
    """


def json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
