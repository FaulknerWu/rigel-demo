"""Rigel Web 演示后端。"""

from __future__ import annotations

import json
from json import JSONDecodeError
from pathlib import Path
from threading import Lock
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from rigel_demo.cli import (
    WorkspacePaths,
    WEB_STATIC_DIRECTORY_NAME,
    database_artifact_exists,
    index_repository_workspace,
    index_result_payload,
)
from rigel_demo.graphrag import (
    RigelChatService,
    RigelGraphRAGError,
    build_graphrag_chat_service,
)
from rigel_demo.query.service import (
    DEFAULT_GRAPH_LIMIT,
    RigelGraphReader,
)
from rigel_demo.embedding import (
    EmbeddingConfig,
    RigelEmbedding,
)
from rigel_demo.llm import (
    LLMMessage,
)


def create_app(
    repository_path: Path | None = None,
    *,
    chat_client: RigelChatService | None = None,
    embedding_client: RigelEmbedding | None = None,
) -> FastAPI:
    """创建基于当前仓库 `.rigel` 目录的 Web 演示应用。"""

    workspace_paths = WorkspacePaths.from_repository(repository_path)
    resolved_repository_path = workspace_paths.repository_path
    workspace_path = workspace_paths.workspace_path
    database_path = workspace_paths.database_path
    state_path = workspace_paths.state_path
    static_frontend_path = workspace_path / WEB_STATIC_DIRECTORY_NAME
    static_assets_path = static_frontend_path / "assets"
    static_index_path = static_frontend_path / "index.html"
    graph_name = _read_graph_name(state_path)
    graph_reader = RigelGraphReader(database_path=database_path, graph_name=graph_name)
    active_embedding_client = _resolve_embedding_client(resolved_repository_path, embedding_client)
    active_chat_client = _resolve_chat_client(
        resolved_repository_path,
        chat_client,
        graph_name=graph_name,
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
            chat_reply = active_chat_client.send_messages(messages)
        except RigelGraphRAGError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

        return {
            "status": "success",
            "message": {"role": "assistant", "content": chat_reply.content},
            "queries": [
                {"name": trace.name, "args": trace.args}
                for trace in chat_reply.traces
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

    @field_validator("content")
    @classmethod
    def validate_content(cls, content: str) -> str:
        stripped_content = content.strip()
        if not stripped_content:
            raise ValueError("消息内容不能为空")
        return stripped_content


class ChatRequest(BaseModel):
    """聊天请求体。"""

    messages: list[ChatMessagePayload] = Field(min_length=1, max_length=50)


class WorkspaceStateError(ValueError):
    """Rigel 工作区索引状态文件无效。"""


def _read_graph_name(state_path: Path) -> str:
    """读取索引状态中的图名称。"""

    if not state_path.exists():
        raise WorkspaceStateError(f"未找到索引状态文件，请先执行 rigel index：{state_path}")

    try:
        state_document = json.loads(state_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise WorkspaceStateError(f"读取索引状态文件失败：{state_path}") from error
    except JSONDecodeError as error:
        raise WorkspaceStateError(f"索引状态文件不是合法 JSON：{state_path}") from error

    if not isinstance(state_document, dict):
        raise WorkspaceStateError("索引状态文件根节点必须是 JSON 对象")

    graph_name = state_document.get("graph_name")
    if not isinstance(graph_name, str) or not graph_name.strip():
        raise WorkspaceStateError("索引状态文件必须包含非空字符串字段：graph_name")
    return graph_name.strip()


def _ensure_database_exists(database_path: Path) -> None:
    """确保当前目录已存在 Rigel 本地图数据库。"""

    if not database_artifact_exists(database_path):
        raise HTTPException(
            status_code=404,
            detail=f"未找到数据库文件，请先在目标仓库执行 rigel index: {database_path}",
        )


def _resolve_chat_client(
    repository_path: Path,
    provided_client: RigelChatService | None,
    *,
    graph_name: str,
) -> RigelChatService:
    if provided_client is not None:
        return provided_client

    return build_graphrag_chat_service(
        repository_path=repository_path,
        graph_name=graph_name,
    )


def _resolve_embedding_client(
    repository_path: Path,
    provided_client: RigelEmbedding | None,
) -> RigelEmbedding:
    if provided_client is not None:
        return provided_client

    return RigelEmbedding(EmbeddingConfig.from_repository(repository_path))


def _llm_status(llm_client: RigelChatService) -> dict[str, object]:
    config = llm_client.config
    return {
        "configured": True,
        "runtime": "graphrag-sdk",
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
    ]
