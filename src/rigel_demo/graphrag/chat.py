"""基于 GraphRAG-SDK 的 Rigel chat service。"""

from __future__ import annotations

from collections.abc import Sequence
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
    ) -> None:
        self.config = config
        self._chat_session = _build_chat_session(
            config=config,
            graphrag_config=graphrag_config,
            graph_name=graph_name,
        )

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


def build_graphrag_chat_service(
    *,
    repository_path: Path,
    graph_name: str,
) -> GraphRAGSDKChatService:
    llm_config = LLMConfig.from_repository(repository_path, LLMConfigSection.CHAT)
    graphrag_config = GraphRAGConfig.from_repository(repository_path)
    return GraphRAGSDKChatService(
        config=llm_config,
        graphrag_config=graphrag_config,
        graph_name=graph_name,
    )


def _build_chat_session(
    *,
    config: LLMConfig,
    graphrag_config: GraphRAGConfig,
    graph_name: str,
) -> Any:
    from graphrag_sdk import KnowledgeGraph
    from graphrag_sdk.model_config import KnowledgeGraphModelConfig
    from graphrag_sdk.models.litellm import LiteModel

    ontology = build_rigel_ontology(config=graphrag_config, graph_name=graph_name)
    model = LiteModel(model_name=_litellm_model_name(config))
    model_config = KnowledgeGraphModelConfig.with_model(model)
    kg = KnowledgeGraph(
        name=graph_name,
        model_config=model_config,
        ontology=ontology,
        host=graphrag_config.host,
        port=graphrag_config.port,
        username=graphrag_config.username,
        password=graphrag_config.password,
        cypher_system_instruction=RIGEL_CYPHER_SYSTEM_INSTRUCTION,
        qa_system_instruction=f"{config.system_prompt}\n\n{RIGEL_QA_SYSTEM_INSTRUCTION}",
        cypher_gen_prompt=RIGEL_CYPHER_GENERATION_PROMPT,
        cypher_gen_prompt_history=RIGEL_CYPHER_GENERATION_PROMPT_WITH_HISTORY,
        qa_prompt=RIGEL_QA_PROMPT,
    )
    return kg.chat_session()


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
