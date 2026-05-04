"""Rigel Web 演示后端。"""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Literal, cast

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rigel_demo.cli import (
    FALKORDB_DATABASE_FILE_NAME,
    RIGEL_WORKSPACE_DIRECTORY_NAME,
    WEB_STATIC_DIRECTORY_NAME,
    WORKSPACE_STATE_FILE_NAME,
    database_artifact_exists,
    index_repository_workspace,
    index_result_payload,
)
from rigel_demo.agent.service import (
    DEFAULT_GRAPH_LIMIT,
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
from rigel_demo.agent.langgraph_agent import (
    CodeGraphAgentError,
    CodeGraphAgentRunner,
    LangGraphCodeAgent,
)
from rigel_demo.embedding import (
    EmbeddingConfig,
    EmbeddingRequestError,
    EmbeddingResponseError,
    RigelEmbedding,
)
from rigel_demo.llm import (
    LLMConfig,
    LLMConfigSection,
    LLMMessage,
    LLMResponseError,
)

MAX_AGENT_RECALL_LIMIT = 20
MAX_AGENT_EXPANSION_LIMIT = 10
MAX_AGENT_SOURCE_SLICE_LINES = 300
DEFAULT_AGENT_EXPAND_LIMIT = 20
MAX_AGENT_EXPAND_LIMIT = 100


def create_app(
    repository_path: Path | None = None,
    *,
    chat_client: CodeGraphAgentRunner | None = None,
    embedding_client: RigelEmbedding | None = None,
) -> FastAPI:
    """创建基于当前仓库 `.rigel` 目录的 Web 演示应用。"""

    resolved_repository_path = (repository_path or Path.cwd()).resolve()
    workspace_path = resolved_repository_path / RIGEL_WORKSPACE_DIRECTORY_NAME
    database_path = workspace_path / FALKORDB_DATABASE_FILE_NAME
    state_path = workspace_path / WORKSPACE_STATE_FILE_NAME
    static_frontend_path = workspace_path / WEB_STATIC_DIRECTORY_NAME
    static_assets_path = static_frontend_path / "assets"
    static_index_path = static_frontend_path / "index.html"
    graph_name = _read_graph_name(state_path)
    graph_reader = RigelGraphReader(database_path=database_path, graph_name=graph_name)
    source_reader = RepositorySourceReader(resolved_repository_path)
    active_embedding_client = _resolve_embedding_client(resolved_repository_path, embedding_client)
    active_chat_client = _resolve_chat_client(
        resolved_repository_path,
        chat_client,
        graph_reader=graph_reader,
        source_reader=source_reader,
        embedding_client=active_embedding_client,
    )
    incremental_index_lock = Lock()

    app = FastAPI(title="Rigel Demo", version="0.1.0")

    app.mount("/assets", StaticFiles(directory=static_assets_path), name="assets")

    @app.get("/api/health")
    def health() -> dict[str, object]:
        """返回演示后端与本地数据库状态。"""

        return {
            "status": "ok" if database_artifact_exists(database_path) else "missing_database",
            "repository_path": str(resolved_repository_path),
            "workspace_path": str(workspace_path),
            "database_path": str(database_path),
            "graph_name": graph_name,
            "chat": _llm_status(active_chat_client),
            "embedding": _embedding_status(active_embedding_client),
        }

    @app.get("/api/summary")
    def summary() -> dict[str, object]:
        """返回图谱概览统计。"""

        _ensure_database_exists(database_path)
        return {"status": "success", "summary": graph_reader.summary()}

    @app.get("/api/graph")
    def graph(limit: int = DEFAULT_GRAPH_LIMIT) -> dict[str, object]:
        """返回前端可直接渲染的节点和边。"""

        _ensure_database_exists(database_path)
        if limit < 1:
            raise HTTPException(status_code=400, detail="limit 必须大于 0")
        return {"status": "success", "graph": graph_reader.graph(limit=limit)}

    @app.post("/api/tools/recall")
    def tool_recall(request: AgentSemanticRecallRequest) -> dict[str, object]:
        """工具：基于自然语言问题召回图谱证据上下文。"""

        _ensure_database_exists(database_path)
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="query 不能为空")
        _ensure_agent_range(request.limit, "limit", minimum=1, maximum=MAX_AGENT_RECALL_LIMIT)
        _ensure_agent_range(request.expansion_limit, "expansion_limit", minimum=0, maximum=MAX_AGENT_EXPANSION_LIMIT)
        try:
            query_embedding = active_embedding_client.embed_query(query)
        except (EmbeddingRequestError, EmbeddingResponseError) as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

        return {
            "status": "success",
            "context": graph_reader.context(
                query=query,
                query_embedding=query_embedding,
                embedding_model=active_embedding_client.config.model,
                limit=request.limit,
                expansion_limit=request.expansion_limit,
                source_reader=source_reader,
            ),
        }

    @app.post("/api/tools/anchors")
    def tool_anchors(request: AgentNodeAnchorsRequest) -> dict[str, object]:
        """工具：读取指定图谱节点的源码锚点。"""

        _ensure_database_exists(database_path)
        result = graph_reader.anchors_for_node(request.node_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"未找到节点：{request.node_id}")
        return {"status": "success", **result}

    @app.post("/api/tools/source")
    def tool_source(request: AgentSourceSliceRequest) -> dict[str, object]:
        """工具：读取已索引源码文件的安全行号切片。"""

        _ensure_database_exists(database_path)
        _ensure_agent_range(request.max_lines, "max_lines", minimum=1, maximum=MAX_AGENT_SOURCE_SLICE_LINES)
        try:
            normalized_path = source_reader.normalize_relative_path(request.path)
        except SourcePathError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        source_file = graph_reader.source_file(normalized_path)
        if source_file is None:
            raise HTTPException(status_code=404, detail=f"未找到已索引源码文件：{normalized_path}")

        try:
            source_slice = source_reader.read_slice(
                normalized_path,
                start_line=request.start_line,
                end_line=request.end_line,
                max_lines=request.max_lines,
            )
        except SourceLineRangeError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except SourceFileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except SourceReadError as error:
            raise HTTPException(status_code=500, detail=str(error)) from error

        return {
            "status": "success",
            "source": {
                **source_slice,
                "source_file": source_file,
            },
        }

    @app.post("/api/tools/expand")
    def tool_expand(request: AgentExpandGraphRequest) -> dict[str, object]:
        """工具：读取指定节点的一跳局部图关系。"""

        _ensure_database_exists(database_path)
        _ensure_agent_range(request.limit, "limit", minimum=1, maximum=MAX_AGENT_EXPAND_LIMIT)
        _ensure_agent_edge_types(request.edge_types)
        if graph_reader.node_by_id(request.node_id) is None:
            raise HTTPException(status_code=404, detail=f"未找到节点：{request.node_id}")

        return {
            "status": "success",
            "graph": graph_reader.expand_graph(
                node_id=request.node_id,
                direction=request.direction,
                edge_types=request.edge_types,
                limit=request.limit,
            ),
        }

    @app.post("/api/index/incremental")
    def incremental_index() -> dict[str, object]:
        """执行演示级增量索引并更新当前图数据库。"""

        _ensure_database_exists(database_path)
        if not incremental_index_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="增量索引正在执行，请稍后再试")

        try:
            result = index_repository_workspace(resolved_repository_path, incremental=True)
        except Exception as error:
            raise HTTPException(status_code=500, detail=f"增量索引失败：{error}") from error
        finally:
            incremental_index_lock.release()

        return {
            "status": "success",
            "result": index_result_payload(result),
        }

    @app.post("/api/chat")
    def chat(request: ChatRequest) -> dict[str, object]:
        """调用已配置的 LLM 生成对话回复。"""

        _ensure_database_exists(database_path)

        messages = _to_llm_messages(request.messages)
        if not messages:
            raise HTTPException(status_code=400, detail="消息不能为空")
        if messages[-1].role != "user":
            raise HTTPException(status_code=400, detail="最后一条消息必须来自用户")

        try:
            agent_reply = active_chat_client.generate_reply(messages)
        except (EmbeddingRequestError, EmbeddingResponseError) as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        except CodeGraphAgentError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        except LLMResponseError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

        return {
            "status": "success",
            "message": {"role": "assistant", "content": agent_reply.content},
            "tool_calls": [
                {"name": tool_call.name, "args": tool_call.args}
                for tool_call in agent_reply.tool_calls
            ],
            "model": active_chat_client.config.model,
            "provider": active_chat_client.config.provider,
        }

    @app.get("/", response_model=None)
    def index() -> FileResponse:
        """返回已构建前端。"""

        return FileResponse(static_index_path)

    return app


class ChatMessagePayload(BaseModel):
    """前端传入的聊天消息。"""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    """聊天请求体。"""

    messages: list[ChatMessagePayload] = Field(min_length=1, max_length=50)


class AgentSemanticRecallRequest(BaseModel):
    """Agent 语义召回工具请求体。"""

    query: str = Field(min_length=1)
    limit: int = DEFAULT_RECALL_LIMIT
    expansion_limit: int = DEFAULT_RECALL_EXPANSION_LIMIT


class AgentNodeAnchorsRequest(BaseModel):
    """Agent 节点锚点工具请求体。"""

    node_id: str = Field(min_length=1)


class AgentSourceSliceRequest(BaseModel):
    """Agent 源码切片工具请求体。"""

    path: str = Field(min_length=1)
    start_line: int
    end_line: int
    max_lines: int = DEFAULT_SOURCE_SLICE_MAX_LINES


class AgentExpandGraphRequest(BaseModel):
    """Agent 局部图扩展工具请求体。"""

    node_id: str = Field(min_length=1)
    direction: GraphExpansionDirection = "both"
    edge_types: list[str] = Field(default_factory=lambda: list(VISIBLE_EDGE_TYPES))
    limit: int = DEFAULT_AGENT_EXPAND_LIMIT


def _read_graph_name(state_path: Path) -> str:
    """读取索引状态中的图名称。"""

    return cast(str, json.loads(state_path.read_text(encoding="utf-8"))["graph_name"])


def _ensure_database_exists(database_path: Path) -> None:
    """确保当前目录已存在 Rigel 本地图数据库。"""

    if not database_artifact_exists(database_path):
        raise HTTPException(
            status_code=404,
            detail=f"未找到数据库文件，请先在目标仓库执行 rigel index: {database_path}",
        )


def _ensure_agent_range(value: int, name: str, *, minimum: int, maximum: int) -> None:
    """校验 Agent 工具数值参数，保持工具错误统一返回 400。"""

    if isinstance(value, bool) or value < minimum or value > maximum:
        raise HTTPException(status_code=400, detail=f"{name} 必须在 {minimum} 到 {maximum} 之间")


def _ensure_agent_edge_types(edge_types: list[str]) -> None:
    """校验 Agent 局部图扩展的边类型。"""

    invalid_edge_types = [edge_type for edge_type in edge_types if edge_type not in VISIBLE_EDGE_TYPES]
    if invalid_edge_types:
        raise HTTPException(status_code=400, detail=f"不支持的 edge_types：{', '.join(invalid_edge_types)}")


def _resolve_chat_client(
    repository_path: Path,
    provided_client: CodeGraphAgentRunner | None,
    *,
    graph_reader: RigelGraphReader,
    source_reader: RepositorySourceReader,
    embedding_client: RigelEmbedding,
) -> CodeGraphAgentRunner:
    if provided_client is not None:
        return provided_client

    return LangGraphCodeAgent(
        config=LLMConfig.from_repository(repository_path, LLMConfigSection.CHAT),
        graph_reader=graph_reader,
        source_reader=source_reader,
        embedding_client=embedding_client,
    )


def _resolve_embedding_client(
    repository_path: Path,
    provided_client: RigelEmbedding | None,
) -> RigelEmbedding:
    if provided_client is not None:
        return provided_client

    return RigelEmbedding(EmbeddingConfig.from_repository(repository_path))


def _llm_status(llm_client: CodeGraphAgentRunner) -> dict[str, object]:
    config = llm_client.config
    return {
        "configured": True,
        "provider": config.provider,
        "model": config.model,
        "base_url": config.base_url,
    }


def _embedding_status(embedding_client: RigelEmbedding) -> dict[str, object]:
    config = embedding_client.config
    return {
        "configured": True,
        "provider": config.provider,
        "format": config.format.value,
        "model": config.model,
        "base_url": config.base_url,
        "dimensions": config.dimensions,
        "batch_size": config.batch_size,
    }


def _to_llm_messages(messages: list[ChatMessagePayload]) -> list[LLMMessage]:
    return [
        LLMMessage(role=message.role, content=message.content)
        for message in messages
        if message.content.strip()
    ]
