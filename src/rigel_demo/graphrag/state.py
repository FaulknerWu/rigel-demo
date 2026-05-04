"""LangGraph GraphRAG 状态类型。"""

from __future__ import annotations

from typing import Any, TypedDict

from rigel_demo.graphrag.models import GraphRAGTrace
from rigel_demo.llm import LLMMessage


class ToolCall(TypedDict):
    id: str
    name: str
    args: dict[str, object]


class GraphRAGState(TypedDict):
    messages: list[LLMMessage]
    question: str
    last_answer: str | None
    schema: str
    agent_messages: list[Any]
    pending_tool_calls: list[ToolCall]
    evidence: list[dict[str, object]]
    answer: str
    traces: list[GraphRAGTrace]
    known_node_ids: list[str]
    visited_node_ids: list[str]
    tool_call_count: int
    tool_counts: dict[str, int]
    retrieval_complete: bool
    limit_reached: bool
