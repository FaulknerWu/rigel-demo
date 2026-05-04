"""基于 LangChain/LangGraph 的代码图谱工具调度 Agent。"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from rigel_demo.agent.service import (
    DEFAULT_RECALL_EXPANSION_LIMIT,
    DEFAULT_RECALL_LIMIT,
    DEFAULT_SOURCE_SLICE_MAX_LINES,
    VISIBLE_EDGE_TYPES,
    GraphExpansionDirection,
    RepositorySourceReader,
    RigelGraphReader,
    SourceFileNotFoundError,
    SourceLineRangeError,
    SourcePathError,
    SourceReadError,
)
from rigel_demo.embedding import RigelEmbedding
from rigel_demo.llm import LLMConfig, LLMMessage, LLMResponseError


MAX_AGENT_RECALL_LIMIT = 20
MAX_AGENT_EXPANSION_LIMIT = 10
MAX_AGENT_SOURCE_SLICE_LINES = 300
DEFAULT_AGENT_EXPAND_LIMIT = 20
MAX_AGENT_EXPAND_LIMIT = 100
AGENT_RECURSION_LIMIT = 12
AGENT_SYSTEM_PROMPT_SUFFIX = (
    "\n\n你可以自由调用代码图谱工具获取事实证据。回答代码问题前优先调用 recall 查找相关节点；"
    "需要源码证据时调用 source，已知节点 ID 时可调用 anchors 或 expand。回答必须基于工具返回的事实，"
    "尽量引用节点名、文件路径与行号；无法从工具结果确认时明确说明不确定。"
)


class CodeGraphAgentError(RuntimeError):
    """代码图谱 Agent 执行失败。"""


@dataclass(frozen=True, slots=True)
class ToolCallTrace:
    """前端展示用工具调用轨迹。"""

    name: str
    args: dict[str, object]


@dataclass(frozen=True, slots=True)
class AgentReply:
    """Agent 生成结果。"""

    content: str
    tool_calls: list[ToolCallTrace]


class CodeGraphAgentRunner(Protocol):
    config: LLMConfig

    def generate_reply(self, messages: Sequence[LLMMessage]) -> AgentReply:
        """根据对话历史生成回答并返回工具调用轨迹。"""


class RecallToolInput(BaseModel):
    """语义召回工具输入。"""

    query: str = Field(min_length=1, description="用户问题或要检索的代码概念")
    limit: int = Field(default=DEFAULT_RECALL_LIMIT, ge=1, le=MAX_AGENT_RECALL_LIMIT)
    expansion_limit: int = Field(default=DEFAULT_RECALL_EXPANSION_LIMIT, ge=0, le=MAX_AGENT_EXPANSION_LIMIT)


class AnchorsToolInput(BaseModel):
    """源码锚点工具输入。"""

    node_id: str = Field(min_length=1, description="图谱节点 ID")


class SourceToolInput(BaseModel):
    """源码切片工具输入。"""

    path: str = Field(min_length=1, description="仓库内源码相对路径")
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    max_lines: int = Field(default=DEFAULT_SOURCE_SLICE_MAX_LINES, ge=1, le=MAX_AGENT_SOURCE_SLICE_LINES)


class ExpandToolInput(BaseModel):
    """局部图扩展工具输入。"""

    node_id: str = Field(min_length=1, description="图谱节点 ID")
    direction: GraphExpansionDirection = "both"
    edge_types: list[str] = Field(default_factory=lambda: list(VISIBLE_EDGE_TYPES))
    limit: int = Field(default=DEFAULT_AGENT_EXPAND_LIMIT, ge=1, le=MAX_AGENT_EXPAND_LIMIT)


class LangGraphCodeAgent:
    """用 LangChain Agent 封装 LangGraph 工具调用循环。"""

    def __init__(
        self,
        *,
        config: LLMConfig,
        graph_reader: RigelGraphReader,
        source_reader: RepositorySourceReader,
        embedding_client: RigelEmbedding,
    ) -> None:
        self.config = config
        self._tools = _build_code_graph_tools(
            graph_reader=graph_reader,
            source_reader=source_reader,
            embedding_client=embedding_client,
        )
        self._agent = create_agent(
            model=_chat_model_from_config(config),
            tools=self._tools,
            system_prompt=f"{config.system_prompt}{AGENT_SYSTEM_PROMPT_SUFFIX}",
        )

    def generate_reply(self, messages: Sequence[LLMMessage]) -> AgentReply:
        langchain_messages = _to_langchain_messages(messages)
        if not langchain_messages:
            raise LLMResponseError("消息列表不能为空")

        try:
            result = self._agent.invoke(
                {"messages": langchain_messages},
                config={"recursion_limit": AGENT_RECURSION_LIMIT},
            )
        except Exception as error:
            raise CodeGraphAgentError(f"Agent 调用失败：{error}") from error

        result_messages = cast(list[BaseMessage], result.get("messages", []))
        content = _last_assistant_content(result_messages)
        return AgentReply(content=content, tool_calls=_tool_call_traces(result_messages))


def _build_code_graph_tools(
    *,
    graph_reader: RigelGraphReader,
    source_reader: RepositorySourceReader,
    embedding_client: RigelEmbedding,
) -> list[StructuredTool]:
    def recall(query: str, limit: int = DEFAULT_RECALL_LIMIT, expansion_limit: int = DEFAULT_RECALL_EXPANSION_LIMIT) -> str:
        """按自然语言问题召回代码图谱证据、相关节点、源码锚点和少量源码切片。"""

        query_text = query.strip()
        if not query_text:
            return _tool_error("query 不能为空")
        query_embedding = embedding_client.embed_query(query_text)
        return _tool_json(
            {
                "status": "success",
                "context": graph_reader.context(
                    query=query_text,
                    query_embedding=query_embedding,
                    embedding_model=embedding_client.config.model,
                    limit=limit,
                    expansion_limit=expansion_limit,
                    source_reader=source_reader,
                ),
            }
        )

    def anchors(node_id: str) -> str:
        """读取指定图谱节点的源码文件和源码锚点坐标。"""

        result = graph_reader.anchors_for_node(node_id)
        if result is None:
            return _tool_error(f"未找到节点：{node_id}")
        return _tool_json({"status": "success", **result})

    def source(path: str, start_line: int, end_line: int, max_lines: int = DEFAULT_SOURCE_SLICE_MAX_LINES) -> str:
        """读取已索引源码文件的安全行号切片。"""

        try:
            normalized_path = source_reader.normalize_relative_path(path)
            source_file = graph_reader.source_file(normalized_path)
            if source_file is None:
                return _tool_error(f"未找到已索引源码文件：{normalized_path}")
            source_slice = source_reader.read_slice(
                normalized_path,
                start_line=start_line,
                end_line=end_line,
                max_lines=max_lines,
            )
        except (SourcePathError, SourceLineRangeError, SourceFileNotFoundError, SourceReadError) as error:
            return _tool_error(str(error))
        return _tool_json({"status": "success", "source": {**source_slice, "source_file": source_file}})

    def expand(
        node_id: str,
        direction: GraphExpansionDirection = "both",
        edge_types: list[str] | None = None,
        limit: int = DEFAULT_AGENT_EXPAND_LIMIT,
    ) -> str:
        """读取指定图谱节点的一跳局部关系。"""

        normalized_edge_types = edge_types or list(VISIBLE_EDGE_TYPES)
        invalid_edge_types = [edge_type for edge_type in normalized_edge_types if edge_type not in VISIBLE_EDGE_TYPES]
        if invalid_edge_types:
            return _tool_error(f"不支持的 edge_types：{', '.join(invalid_edge_types)}")
        if graph_reader.node_by_id(node_id) is None:
            return _tool_error(f"未找到节点：{node_id}")
        return _tool_json(
            {
                "status": "success",
                "graph": graph_reader.expand_graph(
                    node_id=node_id,
                    direction=direction,
                    edge_types=normalized_edge_types,
                    limit=limit,
                ),
            }
        )

    return [
        StructuredTool.from_function(
            recall,
            name="recall",
            description=recall.__doc__ or "",
            args_schema=RecallToolInput,
        ),
        StructuredTool.from_function(
            anchors,
            name="anchors",
            description=anchors.__doc__ or "",
            args_schema=AnchorsToolInput,
        ),
        StructuredTool.from_function(
            source,
            name="source",
            description=source.__doc__ or "",
            args_schema=SourceToolInput,
        ),
        StructuredTool.from_function(
            expand,
            name="expand",
            description=expand.__doc__ or "",
            args_schema=ExpandToolInput,
        ),
    ]


def _chat_model_from_config(config: LLMConfig) -> ChatOpenAI:
    return ChatOpenAI(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        timeout=config.timeout_seconds,
        temperature=config.temperature,
        max_completion_tokens=config.max_output_tokens,
    )


def _to_langchain_messages(messages: Sequence[LLMMessage]) -> list[BaseMessage]:
    langchain_messages: list[BaseMessage] = []
    for message in messages:
        content = message.content.strip()
        if not content:
            continue
        if message.role == "user":
            langchain_messages.append(HumanMessage(content=content))
        else:
            langchain_messages.append(AIMessage(content=content))
    return langchain_messages


def _last_assistant_content(messages: Sequence[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            content = message.content
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                text_parts = [
                    str(part.get("text", "")).strip()
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                text = "\n".join(part for part in text_parts if part)
                if text:
                    return text
    raise LLMResponseError("Agent 未返回可展示文本")


def _tool_call_traces(messages: Sequence[BaseMessage]) -> list[ToolCallTrace]:
    traces: list[ToolCallTrace] = []
    for message in messages:
        tool_calls = getattr(message, "tool_calls", None)
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            name = tool_call.get("name")
            args = tool_call.get("args")
            if isinstance(name, str) and isinstance(args, dict):
                traces.append(ToolCallTrace(name=name, args=dict(args)))
    return traces


def _tool_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _tool_error(message: str) -> str:
    return _tool_json({"status": "error", "error": message})
