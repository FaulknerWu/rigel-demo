"""LangGraph GraphRAG workflow 定义。"""

from __future__ import annotations

from typing import Any

from rigel_demo.graphrag.state import GraphRAGState


def build_workflow(service: Any) -> Any:
    from langgraph.graph import END, StateGraph

    workflow = StateGraph(GraphRAGState)
    workflow.add_node("prepare_question", service._prepare_question)
    workflow.add_node("load_schema", service._load_schema)
    workflow.add_node("plan_tool_calls", service._plan_tool_calls)
    workflow.add_node("execute_tools", service._execute_tools)
    workflow.add_node("generate_answer", service._generate_answer)
    workflow.set_entry_point("prepare_question")
    workflow.add_edge("prepare_question", "load_schema")
    workflow.add_edge("load_schema", "plan_tool_calls")
    workflow.add_conditional_edges(
        "plan_tool_calls",
        route_after_planning,
        {
            "tools": "execute_tools",
            "answer": "generate_answer",
        },
    )
    workflow.add_conditional_edges(
        "execute_tools",
        route_after_tools,
        {
            "plan": "plan_tool_calls",
            "answer": "generate_answer",
        },
    )
    workflow.add_edge("generate_answer", END)
    return workflow.compile()


def route_after_planning(state: GraphRAGState) -> str:
    if state.get("pending_tool_calls"):
        return "tools"
    return "answer"


def route_after_tools(state: GraphRAGState) -> str:
    if state.get("retrieval_complete") or state.get("limit_reached"):
        return "answer"
    return "plan"
