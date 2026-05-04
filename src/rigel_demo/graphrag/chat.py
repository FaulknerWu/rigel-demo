"""基于 LangGraph 工具编排的 Rigel GraphRAG chat service。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from rigel_demo.config import GraphRAGConfig
from rigel_demo.config.embedding import EmbeddingConfig
from rigel_demo.embedding import RigelEmbedding, build_rigel_embedding
from rigel_demo.graphrag.models import (
    GraphRAGReply,
    GraphRAGTrace,
    RigelChatService,
    RigelGraphRAGError,
)
from rigel_demo.graphrag.runtime import (
    build_falkordb_graph,
    runtime_graphrag_config,
    start_embedded_falkordb_runtime,
)
from rigel_demo.graphrag.state import GraphRAGState
from rigel_demo.graphrag.tools import (
    DEFAULT_EXPAND_LIMIT_PER_NODE,
    DEFAULT_RELATION_LIMIT_PER_NODE,
    DEFAULT_VECTOR_SEARCH_LIMIT,
    MAX_EXPAND_LIMIT_PER_NODE,
    MAX_RELATION_LIMIT_PER_NODE,
    MAX_TOTAL_TOOL_CALLS,
    MAX_VECTOR_SEARCH_LIMIT,
    TOOL_CALL_LIMITS,
    TOOL_EDGE_TYPES,
    TOOL_NAMES,
    agent_messages_with_evidence,
    bind_tool_schemas,
    bounded_limit,
    direction_arg,
    edge_types_arg,
    evidence_prompt_payload,
    expand_graph_query,
    extract_tool_calls,
    filter_known_node_ids,
    filter_known_unvisited_node_ids,
    initial_agent_messages,
    query_graph,
    relation_from_row,
    rigel_schema_instruction,
    seed_row_values,
    string_arg,
    string_list_arg,
    tool_message,
    tool_result,
    trace_from_tool_result,
)
from rigel_demo.graphrag.workflow import build_workflow
from rigel_demo.llm import (
    LLMConfig,
    LLMConfigSection,
    LLMMessage,
    build_langchain_chat_model,
    extract_message_text,
    normalize_messages,
)
from rigel_demo.project.summaries import RETRIEVAL_SUMMARY_PURPOSE
from rigel_demo.query.presentation import (
    cosine_distance_to_similarity,
    format_node,
    format_summary,
)
from rigel_demo.query.service import GraphExpansionDirection, VISIBLE_NODE_TYPES

# 兼容既有测试与内部调用方，后续若需要可改为直接从 runtime 导入。
_start_embedded_falkordb_runtime = start_embedded_falkordb_runtime


class LangGraphChatService:
    """使用 LangGraph 显式状态机调用受控图谱工具并生成回答。"""

    def __init__(
        self,
        *,
        config: LLMConfig,
        graphrag_config: GraphRAGConfig,
        graph_name: str,
        embedding_client: RigelEmbedding,
        database_path: Path | None = None,
        chat_model: Any | None = None,
        graph: Any | None = None,
    ) -> None:
        self.config = config
        self._graphrag_config = graphrag_config
        self._graph_name = graph_name
        self._embedding_client = embedding_client
        self._embedded_database = (
            start_embedded_falkordb_runtime(database_path=database_path, host=graphrag_config.host)
            if database_path is not None
            else None
        )
        self._chat_model = chat_model or build_langchain_chat_model(config)
        self._tool_model = bind_tool_schemas(self._chat_model)
        self._graph = graph
        self._workflow = build_workflow(self)

    def send_messages(self, messages: Sequence[LLMMessage]) -> GraphRAGReply:
        initial_state = GraphRAGState(
            messages=list(messages),
            question="",
            last_answer=None,
            schema="",
            agent_messages=[],
            pending_tool_calls=[],
            evidence=[],
            answer="",
            traces=[],
            known_node_ids=[],
            visited_node_ids=[],
            tool_call_count=0,
            tool_counts={tool_name: 0 for tool_name in TOOL_NAMES},
            retrieval_complete=False,
            limit_reached=False,
        )
        try:
            result = self._workflow.invoke(initial_state)
        except RigelGraphRAGError:
            raise
        except Exception as error:
            raise RigelGraphRAGError(f"LangGraph GraphRAG 调用失败：{error}") from error

        answer = str(result.get("answer", "")).strip()
        if not answer:
            raise RigelGraphRAGError("LangGraph GraphRAG 未返回可展示文本")
        return GraphRAGReply(content=answer, traces=list(result.get("traces", [])))

    def close(self) -> None:
        if self._embedded_database is not None:
            self._embedded_database.client.close()

    def _prepare_question(self, state: GraphRAGState) -> dict[str, object]:
        messages = normalize_messages(state["messages"])
        if not messages:
            raise RigelGraphRAGError("消息列表不能为空")
        if messages[-1].role != "user":
            raise RigelGraphRAGError("最后一条消息必须来自用户")

        last_answer = next(
            (message.content for message in reversed(messages[:-1]) if message.role == "assistant"),
            None,
        )
        return {
            "messages": messages,
            "question": messages[-1].content,
            "last_answer": last_answer,
        }

    def _load_schema(self, state: GraphRAGState) -> dict[str, object]:
        graph = self._active_graph()
        refresh_schema = getattr(graph, "refresh_schema", None) or getattr(graph, "refreshSchema", None)
        if callable(refresh_schema):
            refresh_schema()
        schema_value = getattr(graph, "get_schema", None)
        if callable(schema_value):
            schema_text = str(schema_value())
        elif isinstance(schema_value, str):
            schema_text = schema_value
        else:
            schema_text = str(schema_value or "")
        schema = rigel_schema_instruction(schema_text)
        return {
            "schema": schema,
            "agent_messages": initial_agent_messages(
                messages=state["messages"],
                schema=schema,
                system_prompt=self.config.system_prompt,
            ),
        }

    def _plan_tool_calls(self, state: GraphRAGState) -> dict[str, object]:
        if state["limit_reached"]:
            return {"pending_tool_calls": [], "retrieval_complete": True}

        response = self._tool_model.invoke(agent_messages_with_evidence(state))
        tool_calls = extract_tool_calls(response)
        return {
            "agent_messages": [*state["agent_messages"], response],
            "pending_tool_calls": tool_calls,
            "retrieval_complete": not tool_calls,
        }

    def _execute_tools(self, state: GraphRAGState) -> dict[str, object]:
        agent_messages = list(state["agent_messages"])
        evidence = list(state["evidence"])
        traces = list(state["traces"])
        known_node_ids = set(state["known_node_ids"])
        visited_node_ids = set(state["visited_node_ids"])
        tool_counts = dict(state["tool_counts"])
        tool_call_count = int(state["tool_call_count"])
        limit_reached = bool(state["limit_reached"])

        for tool_call in state["pending_tool_calls"]:
            tool_name = tool_call["name"]
            tool_call_id = tool_call["id"]
            if tool_name not in TOOL_CALL_LIMITS:
                result = tool_result(tool_name, [], [f"未知工具已忽略：{tool_name}"])
                agent_messages.append(tool_message(result, tool_call_id))
                traces.append(GraphRAGTrace(name=tool_name, args={"warnings": result["warnings"]}))
                continue

            if tool_call_count >= MAX_TOTAL_TOOL_CALLS or tool_counts[tool_name] >= TOOL_CALL_LIMITS[tool_name]:
                warning = f"工具调用次数达到上限，已跳过：{tool_name}"
                result = tool_result(tool_name, [], [warning])
                agent_messages.append(tool_message(result, tool_call_id))
                traces.append(GraphRAGTrace(name=tool_name, args={"warnings": result["warnings"]}))
                limit_reached = True
                break

            tool_call_count += 1
            tool_counts[tool_name] += 1
            result = self._run_tool(
                tool_name,
                tool_call["args"],
                fallback_query_text=state["question"],
                known_node_ids=known_node_ids,
                visited_node_ids=visited_node_ids,
            )
            evidence.extend(cast(list[dict[str, object]], result["items"]))
            traces.append(trace_from_tool_result(tool_name, tool_call["args"], result))
            agent_messages.append(tool_message(result, tool_call_id))

        return {
            "agent_messages": agent_messages,
            "pending_tool_calls": [],
            "evidence": evidence,
            "traces": traces,
            "known_node_ids": sorted(known_node_ids),
            "visited_node_ids": sorted(visited_node_ids),
            "tool_call_count": tool_call_count,
            "tool_counts": tool_counts,
            "limit_reached": limit_reached,
            "retrieval_complete": limit_reached,
        }

    def _run_tool(
        self,
        tool_name: str,
        args: Mapping[str, object],
        *,
        fallback_query_text: str,
        known_node_ids: set[str],
        visited_node_ids: set[str],
    ) -> dict[str, object]:
        match tool_name:
            case "vector_search_seeds":
                return self._vector_search_seeds(args, fallback_query_text=fallback_query_text, known_node_ids=known_node_ids)
            case "expand_neighbors":
                return self._expand_neighbors(args, known_node_ids=known_node_ids, visited_node_ids=visited_node_ids)
            case "query_relation":
                return self._query_relation(args, known_node_ids=known_node_ids)
            case _:
                return tool_result(tool_name, [], [f"未知工具已忽略：{tool_name}"])

    def _vector_search_seeds(
        self,
        args: Mapping[str, object],
        *,
        fallback_query_text: str = "",
        known_node_ids: set[str],
    ) -> dict[str, object]:
        query_text = string_arg(args, "query_text", default=fallback_query_text)
        limit, limit_warnings = bounded_limit(
            args.get("limit"),
            default=DEFAULT_VECTOR_SEARCH_LIMIT,
            maximum=MAX_VECTOR_SEARCH_LIMIT,
            label="limit",
        )
        warnings = list(limit_warnings)
        query_embedding = self._embedding_client.embed_query(query_text)
        rows = query_graph(
            self._active_graph(),
            """
            CALL db.idx.vector.queryNodes('Summary', 'embedding', $vector_limit, vecf32($query_embedding))
            YIELD node AS summary, score AS distance
            MATCH (summary)-[:DESCRIBES]->(target:RigelNode)
            WHERE summary.purpose = $purpose
              AND summary.embedding_model = $embedding_model
              AND summary.embedding_dimensions = $embedding_dimensions
              AND target.rigel_type IN $visible_node_types
            RETURN summary.id AS summary_id, properties(summary) AS summary_properties,
                   target.id AS node_id, properties(target) AS node_properties, distance AS distance
            ORDER BY distance ASC
            """,
            {
                "purpose": RETRIEVAL_SUMMARY_PURPOSE,
                "embedding_model": self._embedding_client.config.model,
                "embedding_dimensions": len(query_embedding),
                "visible_node_types": list(VISIBLE_NODE_TYPES),
                "query_embedding": query_embedding,
                "vector_limit": limit,
            },
        )
        items: list[dict[str, object]] = []
        for row in rows:
            summary_id, summary_properties, node_id, node_properties, distance = seed_row_values(row)
            score = cosine_distance_to_similarity(float(distance))
            if score <= 0:
                continue
            node = format_node(str(node_id), cast(Mapping[str, object], node_properties))
            known_node_ids.add(str(node["id"]))
            items.append(
                {
                    "tool": "vector_search_seeds",
                    "score": score,
                    "summary": format_summary(str(summary_id), cast(Mapping[str, object], summary_properties)),
                    "node": node,
                }
            )

        items.sort(key=lambda item: (-cast(float, item["score"]), str(cast(Mapping[str, object], item["node"])["label"])))
        return tool_result("vector_search_seeds", items[:limit], warnings)

    def _expand_neighbors(
        self,
        args: Mapping[str, object],
        *,
        known_node_ids: set[str],
        visited_node_ids: set[str],
    ) -> dict[str, object]:
        requested_node_ids = string_list_arg(args, "node_ids")
        direction = direction_arg(args)
        edge_types, edge_warnings = edge_types_arg(args.get("edge_types"), default=list(TOOL_EDGE_TYPES))
        limit_per_node, limit_warnings = bounded_limit(
            args.get("limit_per_node"),
            default=DEFAULT_EXPAND_LIMIT_PER_NODE,
            maximum=MAX_EXPAND_LIMIT_PER_NODE,
            label="limit_per_node",
        )
        warnings = [*edge_warnings, *limit_warnings]
        node_ids = filter_known_unvisited_node_ids(
            requested_node_ids,
            known_node_ids=known_node_ids,
            visited_node_ids=visited_node_ids,
            warnings=warnings,
        )
        relations = self._relations_for_node_ids(
            node_ids,
            direction=direction,
            edge_types=edge_types,
            limit_per_node=limit_per_node,
            known_node_ids=known_node_ids,
        )
        visited_node_ids.update(node_ids)
        return tool_result("expand_neighbors", relations, warnings)

    def _query_relation(
        self,
        args: Mapping[str, object],
        *,
        known_node_ids: set[str],
    ) -> dict[str, object]:
        requested_node_ids = string_list_arg(args, "node_ids")
        relation_type = string_arg(args, "relation_type", default="")
        direction = direction_arg(args)
        limit_per_node, limit_warnings = bounded_limit(
            args.get("limit_per_node"),
            default=DEFAULT_RELATION_LIMIT_PER_NODE,
            maximum=MAX_RELATION_LIMIT_PER_NODE,
            label="limit_per_node",
        )
        warnings = list(limit_warnings)
        if relation_type not in TOOL_EDGE_TYPES:
            return tool_result("query_relation", [], [*warnings, f"非法关系类型已忽略：{relation_type}"])
        node_ids = filter_known_node_ids(requested_node_ids, known_node_ids=known_node_ids, warnings=warnings)
        relations = self._relations_for_node_ids(
            node_ids,
            direction=direction,
            edge_types=[relation_type],
            limit_per_node=limit_per_node,
            known_node_ids=known_node_ids,
        )
        return tool_result("query_relation", relations, warnings)

    def _relations_for_node_ids(
        self,
        node_ids: list[str],
        *,
        direction: GraphExpansionDirection,
        edge_types: list[str],
        limit_per_node: int,
        known_node_ids: set[str],
    ) -> list[dict[str, object]]:
        relations: list[dict[str, object]] = []
        for node_id in node_ids:
            rows = query_graph(
                self._active_graph(),
                expand_graph_query(direction),
                {
                    "node_id": node_id,
                    "visible_node_types": list(VISIBLE_NODE_TYPES),
                    "edge_types": edge_types,
                    "limit": limit_per_node,
                },
            )
            for row in rows:
                relation = relation_from_row(row, origin_node_id=node_id)
                known_node_ids.add(str(cast(Mapping[str, object], relation["node"])["id"]))
                relations.append(relation)
        return relations

    def _generate_answer(self, state: GraphRAGState) -> dict[str, object]:
        from rigel_demo.graphrag.prompts import RIGEL_QA_PROMPT, RIGEL_QA_SYSTEM_INSTRUCTION

        if state.get("answer"):
            return {}

        response = self._chat_model.invoke(
            [
                ("system", f"{self.config.system_prompt}\n\n{RIGEL_QA_SYSTEM_INSTRUCTION}"),
                (
                    "user",
                    RIGEL_QA_PROMPT.format(
                        question=state["question"],
                        last_answer=state.get("last_answer") or "",
                        evidence=evidence_prompt_payload(state),
                    ),
                ),
            ]
        )
        answer = extract_message_text(response)
        if not answer:
            raise RigelGraphRAGError("回答模型未返回可展示文本")
        return {"answer": answer}

    def _active_graph(self) -> Any:
        if self._graph is None:
            self._graph = build_falkordb_graph(
                config=runtime_graphrag_config(self._graphrag_config, self._embedded_database),
                graph_name=self._graph_name,
            )
        return self._graph


def build_graphrag_chat_service(
    *,
    repository_path: Path,
    graph_name: str,
    database_path: Path | None = None,
    embedding_client: RigelEmbedding | None = None,
) -> LangGraphChatService:
    llm_config = LLMConfig.from_repository(repository_path, LLMConfigSection.CHAT)
    graphrag_config = GraphRAGConfig.from_repository(repository_path)
    active_embedding_client = embedding_client or build_rigel_embedding(EmbeddingConfig.from_repository(repository_path))
    return LangGraphChatService(
        config=llm_config,
        graphrag_config=graphrag_config,
        graph_name=graph_name,
        database_path=database_path,
        embedding_client=active_embedding_client,
    )
