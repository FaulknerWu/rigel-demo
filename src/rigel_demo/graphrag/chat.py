"""基于 GraphRAG-SDK 的 Rigel chat service。"""

from __future__ import annotations

import os
from collections.abc import Sequence
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from rigel_demo.graphrag.config import GraphRAGConfig
from rigel_demo.graphrag.ontology import build_rigel_ontology
from rigel_demo.graphrag.prompts import (
    RIGEL_CYPHER_GENERATION_PROMPT,
    RIGEL_CYPHER_GENERATION_PROMPT_WITH_HISTORY,
    RIGEL_CYPHER_SYSTEM_INSTRUCTION,
    RIGEL_QA_PROMPT,
    RIGEL_QA_SYSTEM_INSTRUCTION,
)
from rigel_demo.llm import LLMConfig, LLMConfigSection, LLMMessage


class RigelGraphRAGError(RuntimeError):
    """GraphRAG-SDK 调用失败。"""


@dataclass(frozen=True, slots=True)
class GraphRAGTrace:
    """前端展示用 GraphRAG 查询轨迹。"""

    name: str
    args: dict[str, object]


@dataclass(frozen=True, slots=True)
class GraphRAGReply:
    """GraphRAG-SDK 生成结果。"""

    content: str
    traces: list[GraphRAGTrace]


@dataclass(frozen=True, slots=True)
class EmbeddedFalkorDBRuntime:
    """GraphRAG-SDK 访问本地 FalkorDBLite 的运行时连接。"""

    client: Any
    host: str
    port: int


@dataclass(frozen=True, slots=True)
class GraphRAGChatRuntime:
    """GraphRAG chat session 及其依赖的本地数据库运行时。"""

    chat_session: Any
    knowledge_graph: Any | None = None
    embedded_database: EmbeddedFalkorDBRuntime | None = None


class RigelChatService(Protocol):
    config: LLMConfig

    def send_messages(self, messages: Sequence[LLMMessage]) -> GraphRAGReply:
        """按对话历史生成回答。"""


class GraphRAGSDKChatService:
    """通过 KnowledgeGraph.chat_session().send_message() 查询 Rigel 图谱。"""

    def __init__(
        self,
        *,
        config: LLMConfig,
        graphrag_config: GraphRAGConfig,
        graph_name: str,
        database_path: Path | None = None,
    ) -> None:
        self.config = config
        self._environment_snapshot = _apply_litellm_environment(config)
        if database_path is None:
            try:
                self._runtime = GraphRAGChatRuntime(
                    chat_session=_build_chat_session(
                        config=config,
                        graphrag_config=graphrag_config,
                        graph_name=graph_name,
                    )
                )
            except Exception:
                _restore_litellm_environment(self._environment_snapshot)
                raise
        else:
            try:
                self._runtime = _build_chat_runtime(
                    config=config,
                    graphrag_config=graphrag_config,
                    graph_name=graph_name,
                    database_path=database_path,
                )
            except Exception:
                _restore_litellm_environment(self._environment_snapshot)
                raise
        self._chat_session = self._runtime.chat_session

    def send_messages(self, messages: Sequence[LLMMessage]) -> GraphRAGReply:
        if not messages:
            raise RigelGraphRAGError("消息列表不能为空")
        if messages[-1].role != "user":
            raise RigelGraphRAGError("最后一条消息必须来自用户")

        try:
            response = self._send_history(messages)
        except Exception as error:
            raise RigelGraphRAGError(f"GraphRAG-SDK 调用失败：{error}") from error

        return GraphRAGReply(
            content=_response_text(response),
            traces=_response_traces(response),
        )

    def _send_history(self, messages: Sequence[LLMMessage]) -> object:
        response: object = None
        for message in messages:
            if message.role != "user":
                continue
            response = self._chat_session.send_message(message.content)
        if response is None:
            raise RigelGraphRAGError("消息列表中没有用户消息")
        return response

    def close(self) -> None:
        if self._runtime.knowledge_graph is not None:
            self._runtime.knowledge_graph.db.close()
        if self._runtime.embedded_database is not None:
            self._runtime.embedded_database.client.close()
        _restore_litellm_environment(self._environment_snapshot)
        self._environment_snapshot = {}


def build_graphrag_chat_service(
    *,
    repository_path: Path,
    graph_name: str,
    database_path: Path | None = None,
) -> GraphRAGSDKChatService:
    llm_config = LLMConfig.from_repository(repository_path, LLMConfigSection.CHAT)
    graphrag_config = GraphRAGConfig.from_repository(repository_path)
    return GraphRAGSDKChatService(
        config=llm_config,
        graphrag_config=graphrag_config,
        graph_name=graph_name,
        database_path=database_path,
    )


def _build_chat_session(
    *,
    config: LLMConfig,
    graphrag_config: GraphRAGConfig,
    graph_name: str,
) -> Any:
    return _build_chat_runtime(
        config=config,
        graphrag_config=graphrag_config,
        graph_name=graph_name,
        database_path=None,
    ).chat_session


def _build_chat_runtime(
    *,
    config: LLMConfig,
    graphrag_config: GraphRAGConfig,
    graph_name: str,
    database_path: Path | None,
) -> GraphRAGChatRuntime:
    from graphrag_sdk import KnowledgeGraph
    from graphrag_sdk.model_config import KnowledgeGraphModelConfig
    from graphrag_sdk.models.litellm import LiteModel

    embedded_database = _start_embedded_falkordb_runtime(
        database_path=database_path,
        host=graphrag_config.host,
    ) if database_path is not None else None
    active_graphrag_config = _runtime_graphrag_config(graphrag_config, embedded_database)

    ontology = build_rigel_ontology(config=active_graphrag_config, graph_name=graph_name)
    model = LiteModel(
        model_name=_litellm_model_name(config),
        additional_params=_litellm_additional_params(config),
    )
    model_config = KnowledgeGraphModelConfig.with_model(model)
    kg = KnowledgeGraph(
        name=graph_name,
        model_config=model_config,
        ontology=ontology,
        host=active_graphrag_config.host,
        port=active_graphrag_config.port,
        username=active_graphrag_config.username,
        password=active_graphrag_config.password,
        cypher_system_instruction=RIGEL_CYPHER_SYSTEM_INSTRUCTION,
        qa_system_instruction=f"{config.system_prompt}\n\n{RIGEL_QA_SYSTEM_INSTRUCTION}",
        cypher_gen_prompt=RIGEL_CYPHER_GENERATION_PROMPT,
        cypher_gen_prompt_history=RIGEL_CYPHER_GENERATION_PROMPT_WITH_HISTORY,
        qa_prompt=RIGEL_QA_PROMPT,
    )
    return GraphRAGChatRuntime(
        chat_session=kg.chat_session(),
        knowledge_graph=kg,
        embedded_database=embedded_database,
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


def _litellm_additional_params(config: LLMConfig) -> dict[str, object] | None:
    if config.base_url is None:
        return None
    return {"api_base": config.base_url}


def _apply_litellm_environment(config: LLMConfig) -> dict[str, str | None]:
    environment_values = _litellm_environment_values(config)
    snapshot: dict[str, str | None] = {}
    for environment_name, environment_value in environment_values.items():
        snapshot[environment_name] = os.environ.get(environment_name)
        os.environ[environment_name] = environment_value
    return snapshot


def _restore_litellm_environment(snapshot: dict[str, str | None]) -> None:
    for environment_name, previous_value in snapshot.items():
        if previous_value is None:
            os.environ.pop(environment_name, None)
        else:
            os.environ[environment_name] = previous_value


def _litellm_api_key_environment_names(provider: str) -> tuple[str, ...]:
    match provider:
        case "openai":
            return ("OPENAI_API_KEY",)
        case "azure":
            return ("AZURE_API_KEY",)
        case "anthropic":
            return ("ANTHROPIC_API_KEY",)
        case "cohere":
            return ("COHERE_API_KEY",)
        case "openrouter":
            return ("OPENROUTER_API_KEY",)
        case "gemini":
            return ("GOOGLE_API_KEY", "GEMINI_API_KEY")
        case "groq":
            return ("GROQ_API_KEY",)
        case _:
            return (f"{provider.upper().replace('-', '_')}_API_KEY",)


def _litellm_environment_values(config: LLMConfig) -> dict[str, str]:
    values = {
        environment_name: config.api_key
        for environment_name in _litellm_api_key_environment_names(config.provider)
    }
    if config.base_url is None:
        return values

    match config.provider:
        case "openai":
            values["OPENAI_API_BASE"] = config.base_url
        case "azure":
            values["AZURE_API_BASE"] = config.base_url
        case "ollama" | "ollama_chat":
            values["OLLAMA_API_BASE"] = config.base_url
        case _:
            values[f"{config.provider.upper().replace('-', '_')}_API_BASE"] = config.base_url
    return values


def _litellm_model_name(config: LLMConfig) -> str:
    if "/" in config.model:
        return config.model
    return f"{config.provider}/{config.model}"


def _response_text(response: object) -> str:
    if isinstance(response, str):
        return response

    value = _response_text_value(response)
    if isinstance(value, str) and value.strip():
        return value
    raise RigelGraphRAGError("GraphRAG-SDK 响应缺少 response 文本")


def _response_text_value(response: object) -> object:
    for field_name in ("response", "answer", "content"):
        if isinstance(response, dict):
            value = response.get(field_name)
        else:
            value = getattr(response, field_name, None)
        if value:
            return value
    return None


def _response_traces(response: object) -> list[GraphRAGTrace]:
    if not isinstance(response, dict):
        return []
    traces: list[GraphRAGTrace] = []
    cypher = response.get("cypher")
    if isinstance(cypher, str) and cypher.strip():
        traces.append(GraphRAGTrace(name="cypher", args={"query": cypher}))
    context = response.get("context")
    if context is not None:
        traces.append(GraphRAGTrace(name="context", args={"items": _context_size(context)}))
    return traces


def _context_size(context: object) -> int:
    if isinstance(context, list | tuple | set):
        return len(context)
    if isinstance(context, dict):
        return len(context)
    return 1
