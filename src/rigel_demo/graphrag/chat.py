"""基于 LangGraph 的 Rigel GraphRAG chat service。"""

from __future__ import annotations

import re
import socket
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypedDict

from rigel_demo.config import GraphRAGConfig
from rigel_demo.graphrag.prompts import (
    RIGEL_CYPHER_GENERATION_PROMPT,
    RIGEL_CYPHER_GENERATION_PROMPT_WITH_HISTORY,
    RIGEL_CYPHER_SYSTEM_INSTRUCTION,
    RIGEL_QA_PROMPT,
    RIGEL_QA_SYSTEM_INSTRUCTION,
)
from rigel_demo.llm import (
    LLMConfig,
    LLMConfigSection,
    LLMMessage,
    build_langchain_chat_model,
    extract_message_text,
    normalize_messages,
)

SAFE_QUERY_LIMIT = 20
RIGEL_ALLOWED_NODE_TYPES = ("Repository", "Module", "File", "Entity", "Anchor", "Summary")
RIGEL_ALLOWED_RELATIONSHIPS = ("CONTAINS", "DEPENDS_ON", "SPECIALIZES", "ALIASES", "HAS_ANCHOR", "DESCRIBES")
_FORBIDDEN_CYPHER_PATTERN = re.compile(r"\b(CREATE|MERGE|SET|DELETE|REMOVE|DROP|CALL)\b", re.IGNORECASE)
_LIMIT_PATTERN = re.compile(r"\bLIMIT\s+\d+\b", re.IGNORECASE)
_CODE_FENCE_PATTERN = re.compile(r"^```(?:cypher)?\s*|\s*```$", re.IGNORECASE)
_REGEX_MATCH_PATTERN = re.compile(
    r"(?P<expression>[A-Za-z_][\w.]*(?:\([^)]*\))?)\s*=~\s*(?P<quote>['\"])(?P<pattern>.*?)(?P=quote)",
    re.IGNORECASE,
)


class RigelGraphRAGError(RuntimeError):
    """GraphRAG 调用失败。"""


@dataclass(frozen=True, slots=True)
class GraphRAGTrace:
    """前端展示用 GraphRAG 查询轨迹。"""

    name: str
    args: dict[str, object]


@dataclass(frozen=True, slots=True)
class GraphRAGReply:
    """GraphRAG 生成结果。"""

    content: str
    traces: list[GraphRAGTrace]


@dataclass(frozen=True, slots=True)
class EmbeddedFalkorDBRuntime:
    """LangChain 访问本地 FalkorDBLite 的运行时连接。"""

    client: Any
    host: str
    port: int


class GraphRAGState(TypedDict):
    messages: list[LLMMessage]
    question: str
    last_answer: str | None
    schema: str
    cypher: str
    context: list[dict[str, object]]
    answer: str
    traces: list[GraphRAGTrace]


class RigelChatService(Protocol):
    config: LLMConfig

    def send_messages(self, messages: Sequence[LLMMessage]) -> GraphRAGReply:
        """按对话历史生成回答。"""


class LangGraphChatService:
    """使用 LangGraph 显式状态机查询 Rigel 图谱并生成回答。"""

    def __init__(
        self,
        *,
        config: LLMConfig,
        graphrag_config: GraphRAGConfig,
        graph_name: str,
        database_path: Path | None = None,
        chat_model: Any | None = None,
        graph: Any | None = None,
    ) -> None:
        self.config = config
        self._graphrag_config = graphrag_config
        self._graph_name = graph_name
        self._embedded_database = (
            _start_embedded_falkordb_runtime(database_path=database_path, host=graphrag_config.host)
            if database_path is not None
            else None
        )
        self._chat_model = chat_model or build_langchain_chat_model(config)
        self._graph = graph
        self._workflow = _build_workflow(self)

    def send_messages(self, messages: Sequence[LLMMessage]) -> GraphRAGReply:
        initial_state = GraphRAGState(
            messages=list(messages),
            question="",
            last_answer=None,
            schema="",
            cypher="",
            context=[],
            answer="",
            traces=[],
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

    def _load_schema(self, _state: GraphRAGState) -> dict[str, object]:
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
        return {"schema": _rigel_schema_instruction(schema_text)}

    def _generate_cypher(self, state: GraphRAGState) -> dict[str, object]:
        prompt_template = (
            RIGEL_CYPHER_GENERATION_PROMPT_WITH_HISTORY
            if state.get("last_answer")
            else RIGEL_CYPHER_GENERATION_PROMPT
        )
        user_prompt = prompt_template.format(
            question=state["question"],
            last_answer=state.get("last_answer") or "",
        )
        response = self._chat_model.invoke(
            [
                ("system", RIGEL_CYPHER_SYSTEM_INSTRUCTION.format(ontology=state["schema"])),
                (
                    "user",
                    f"{user_prompt}\n\n"
                    "只输出一条 Cypher，不要输出解释或 Markdown。\n"
                    "FalkorDB 不支持 =~ 正则匹配；模糊匹配必须使用 CONTAINS、STARTS WITH 或 ENDS WITH。",
                ),
            ]
        )
        cypher = _strip_cypher_response(extract_message_text(response))
        return {"cypher": cypher}

    def _validate_cypher(self, state: GraphRAGState) -> dict[str, object]:
        cypher = _sanitize_generated_cypher(state["cypher"])
        if _is_safe_readonly_cypher(cypher):
            return {"cypher": cypher}
        return {
            "answer": "无法生成安全只读查询，因此没有执行图数据库查询。",
            "context": [],
            "traces": [GraphRAGTrace(name="cypher", args={"query": cypher, "rejected": True})],
        }

    def _execute_cypher(self, state: GraphRAGState) -> dict[str, object]:
        cypher = _apply_safe_limit(state["cypher"])
        try:
            context = _normalize_query_result(self._active_graph().query(cypher))
        except Exception as error:
            return {
                "cypher": cypher,
                "context": [],
                "traces": [
                    GraphRAGTrace(
                        name="cypher",
                        args={"query": cypher, "error": f"图查询执行失败，已跳过：{error}"},
                    ),
                    GraphRAGTrace(name="context", args={"items": 0}),
                ],
            }
        traces = [
            GraphRAGTrace(name="cypher", args={"query": cypher}),
            GraphRAGTrace(name="context", args={"items": len(context)}),
        ]
        return {"cypher": cypher, "context": context, "traces": traces}

    def _generate_answer(self, state: GraphRAGState) -> dict[str, object]:
        if state.get("answer"):
            return {}

        response = self._chat_model.invoke(
            [
                ("system", f"{self.config.system_prompt}\n\n{RIGEL_QA_SYSTEM_INSTRUCTION}"),
                (
                    "user",
                    RIGEL_QA_PROMPT.format(
                        question=state["question"],
                        context=state["context"],
                        cypher=state["cypher"],
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
            self._graph = _build_falkordb_graph(
                config=_runtime_graphrag_config(self._graphrag_config, self._embedded_database),
                graph_name=self._graph_name,
            )
        return self._graph


def build_graphrag_chat_service(
    *,
    repository_path: Path,
    graph_name: str,
    database_path: Path | None = None,
) -> LangGraphChatService:
    llm_config = LLMConfig.from_repository(repository_path, LLMConfigSection.CHAT)
    graphrag_config = GraphRAGConfig.from_repository(repository_path)
    return LangGraphChatService(
        config=llm_config,
        graphrag_config=graphrag_config,
        graph_name=graph_name,
        database_path=database_path,
    )


def _build_workflow(service: LangGraphChatService) -> Any:
    from langgraph.graph import END, StateGraph

    workflow = StateGraph(GraphRAGState)
    workflow.add_node("prepare_question", service._prepare_question)
    workflow.add_node("load_schema", service._load_schema)
    workflow.add_node("generate_cypher", service._generate_cypher)
    workflow.add_node("validate_cypher", service._validate_cypher)
    workflow.add_node("execute_cypher", service._execute_cypher)
    workflow.add_node("generate_answer", service._generate_answer)
    workflow.set_entry_point("prepare_question")
    workflow.add_edge("prepare_question", "load_schema")
    workflow.add_edge("load_schema", "generate_cypher")
    workflow.add_edge("generate_cypher", "validate_cypher")
    workflow.add_conditional_edges(
        "validate_cypher",
        _route_after_validation,
        {
            "execute": "execute_cypher",
            "answer": "generate_answer",
        },
    )
    workflow.add_edge("execute_cypher", "generate_answer")
    workflow.add_edge("generate_answer", END)
    return workflow.compile()


def _route_after_validation(state: GraphRAGState) -> str:
    if state.get("answer"):
        return "answer"
    return "execute"


def _build_falkordb_graph(*, config: GraphRAGConfig, graph_name: str) -> Any:
    from langchain_community.graphs import FalkorDBGraph

    return FalkorDBGraph(
        database=graph_name,
        host=config.host,
        port=config.port,
        username=config.username or "",
        password=config.password or "",
    )


def _runtime_graphrag_config(
    config: GraphRAGConfig,
    embedded_database: EmbeddedFalkorDBRuntime | None,
) -> GraphRAGConfig:
    if embedded_database is None:
        return config
    return GraphRAGConfig(
        host=embedded_database.host,
        port=embedded_database.port,
        username=None,
        password=None,
    )


def _start_embedded_falkordb_runtime(*, database_path: Path, host: str) -> EmbeddedFalkorDBRuntime:
    from redislite.falkordb_client import FalkorDB

    bind_host = _embedded_bind_host(host)
    port = _reserve_local_port(bind_host)
    client = FalkorDB(
        str(database_path),
        serverconfig={
            "bind": bind_host,
            "port": str(port),
        },
    )
    return EmbeddedFalkorDBRuntime(client=client, host=bind_host, port=port)


def _embedded_bind_host(host: str) -> str:
    if host in {"127.0.0.1", "localhost"}:
        return "127.0.0.1"
    return "127.0.0.1"


def _reserve_local_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((host, 0))
        return int(server_socket.getsockname()[1])


def _rigel_schema_instruction(schema_text: str) -> str:
    return (
        f"{schema_text}\n\n"
        f"Rigel 允许的节点类型：{', '.join(RIGEL_ALLOWED_NODE_TYPES)}。\n"
        f"Rigel 允许的关系类型：{', '.join(RIGEL_ALLOWED_RELATIONSHIPS)}。"
    )


def _strip_cypher_response(response_text: str) -> str:
    stripped_text = response_text.strip()
    stripped_text = _CODE_FENCE_PATTERN.sub("", stripped_text).strip()
    return stripped_text


def _sanitize_generated_cypher(cypher: str) -> str:
    sanitized_cypher = _strip_cypher_response(cypher).strip().rstrip(";").strip()
    sanitized_cypher = _remove_unsupported_path_helpers(sanitized_cypher)
    return _rewrite_regex_matches_for_falkordb(sanitized_cypher)


def _remove_unsupported_path_helpers(cypher: str) -> str:
    sanitized_cypher = re.sub(r"\b(allShortestPaths|shortestPath)\s*\(", "(", cypher, flags=re.IGNORECASE)
    return re.sub(r"\bpath\s*=\s*", "", sanitized_cypher, flags=re.IGNORECASE)


def _rewrite_regex_matches_for_falkordb(cypher: str) -> str:
    """把常见 LLM 正则匹配改写为 FalkorDB 支持的字符串匹配。"""

    def replace_regex_match(match: re.Match[str]) -> str:
        expression = match.group("expression")
        regex_pattern = match.group("pattern")
        literal = _literal_from_simple_contains_regex(regex_pattern)
        if literal is None:
            return match.group(0)
        return f"toLower(coalesce({expression}, '')) CONTAINS '{_cypher_string_literal(literal.lower())}'"

    return _REGEX_MATCH_PATTERN.sub(replace_regex_match, cypher)


def _literal_from_simple_contains_regex(regex_pattern: str) -> str | None:
    normalized_pattern = regex_pattern.strip()
    normalized_pattern = re.sub(r"^\(\?[iI]\)", "", normalized_pattern)
    normalized_pattern = normalized_pattern.removeprefix(".*").removesuffix(".*")
    if not normalized_pattern or re.search(r"(?<!\\)[\\\[\]{}()+?|^$]", normalized_pattern):
        return None
    return normalized_pattern.replace(r"\.", ".").replace(r"\-", "-").replace(r"\_", "_")


def _cypher_string_literal(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _is_safe_readonly_cypher(cypher: str) -> bool:
    if not cypher:
        return False
    if ";" in cypher:
        return False
    if not cypher.upper().startswith("MATCH "):
        return False
    if "=~" in cypher:
        return False
    return _FORBIDDEN_CYPHER_PATTERN.search(cypher) is None


def _apply_safe_limit(cypher: str) -> str:
    stripped_cypher = cypher.strip()
    if _LIMIT_PATTERN.search(stripped_cypher):
        return stripped_cypher
    return f"{stripped_cypher} LIMIT {SAFE_QUERY_LIMIT}"


def _normalize_query_result(result: object) -> list[dict[str, object]]:
    if isinstance(result, list):
        return [_normalize_context_item(item) for item in result]

    data = getattr(result, "data", None)
    if isinstance(data, list):
        return [_normalize_context_item(item) for item in data]

    result_set = getattr(result, "result_set", None)
    if isinstance(result_set, list):
        return [{"row": _normalize_value(row)} for row in result_set]

    return [{"value": _normalize_value(result)}]


def _normalize_context_item(item: object) -> dict[str, object]:
    if isinstance(item, dict):
        return {str(key): _normalize_value(value) for key, value in item.items()}
    return {"value": _normalize_value(item)}


def _normalize_value(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _normalize_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_normalize_value(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    properties = getattr(value, "properties", None)
    if isinstance(properties, dict):
        return {str(key): _normalize_value(item) for key, item in properties.items()}
    return str(value)
