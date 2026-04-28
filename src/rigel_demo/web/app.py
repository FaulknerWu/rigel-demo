"""Rigel Web 演示后端。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rigel_demo.cli import (
    DEFAULT_GRAPH_NAME,
    FALKORDB_DATABASE_FILE_NAME,
    RIGEL_WORKSPACE_DIRECTORY_NAME,
    WEB_STATIC_DIRECTORY_NAME,
    WORKSPACE_STATE_FILE_NAME,
    database_artifact_exists,
)
from rigel_demo.embedding import (
    EmbeddingConfig,
    EmbeddingConfigurationError,
    EmbeddingRequestError,
    EmbeddingResponseError,
    RigelEmbedding,
)
from rigel_demo.indexing.retrieval_summaries import (
    RETRIEVAL_SUMMARY_PURPOSE,
    cosine_similarity,
)
from rigel_demo.llm import (
    LLMConfig,
    LLMConfigurationError,
    LLMMessage,
    LLMRequestError,
    LLMResponseError,
    RigelLLM,
)
from rigel_demo.storage.falkordb_store import FalkorDBConfig, FalkorDBStore

DEFAULT_GRAPH_LIMIT = 500
DEFAULT_SEARCH_LIMIT = 50
DEFAULT_RECALL_LIMIT = 8
DEFAULT_RECALL_SCAN_LIMIT = 2_000
DEFAULT_RECALL_EXPANSION_LIMIT = 3
DEFAULT_CHAT_CONTEXT_LIMIT = 8
VISIBLE_NODE_TYPES = ("Repository", "Module", "File", "Entity")
VISIBLE_EDGE_TYPES = ("CONTAINS", "DEPENDS_ON", "SPECIALIZES", "ALIASES")


def create_app(
    repository_path: Path | None = None,
    *,
    llm_client: RigelLLM | None = None,
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
    # Web 入口复用 CLI 索引状态，确保展示和 `rigel index` 写入的是同一个本地图谱。
    graph_reader = RigelGraphReader(database_path=database_path, graph_name=graph_name)
    active_llm_client, llm_configuration_error = _resolve_llm_client(resolved_repository_path, llm_client)
    active_embedding_client, embedding_configuration_error = _resolve_embedding_client(resolved_repository_path, embedding_client)

    app = FastAPI(title="Rigel Demo", version="0.1.0")

    if static_assets_path.exists():
        # 前端构建产物由 CLI 放入 .rigel，开发阶段不存在时继续暴露 API 和占位页。
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
            "llm": _llm_status(active_llm_client, llm_configuration_error),
            "embedding": _embedding_status(active_embedding_client, embedding_configuration_error),
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

    @app.get("/api/search")
    def search(q: str, limit: int = DEFAULT_SEARCH_LIMIT) -> dict[str, object]:
        """按节点名称、路径或限定名搜索演示图谱。"""

        _ensure_database_exists(database_path)
        if not q.strip():
            raise HTTPException(status_code=400, detail="q 不能为空")
        if limit < 1:
            raise HTTPException(status_code=400, detail="limit 必须大于 0")
        return {"status": "success", "nodes": graph_reader.search(q.strip(), limit=limit)}

    @app.get("/api/recall")
    def recall(q: str, limit: int = DEFAULT_RECALL_LIMIT) -> dict[str, object]:
        """基于 Summary 向量召回代码图谱种子节点。"""

        _ensure_database_exists(database_path)
        if active_embedding_client is None:
            raise HTTPException(status_code=503, detail=embedding_configuration_error or "Embedding 未配置")
        if not q.strip():
            raise HTTPException(status_code=400, detail="q 不能为空")
        if limit < 1:
            raise HTTPException(status_code=400, detail="limit 必须大于 0")
        try:
            query_embedding = active_embedding_client.embed_query(q.strip())
        except (EmbeddingRequestError, EmbeddingResponseError) as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        return {
            "status": "success",
            "results": graph_reader.recall(
                query_embedding,
                embedding_model=active_embedding_client.config.model,
                limit=limit,
                expansion_limit=DEFAULT_RECALL_EXPANSION_LIMIT,
            ),
        }

    @app.post("/api/chat")
    def chat(request: ChatRequest) -> dict[str, object]:
        """调用已配置的 LLM 生成对话回复。"""

        if active_llm_client is None:
            raise HTTPException(status_code=503, detail=llm_configuration_error or "LLM 未配置")
        if database_artifact_exists(database_path) and active_embedding_client is None:
            raise HTTPException(status_code=503, detail=embedding_configuration_error or "Embedding 未配置")

        messages = _to_llm_messages(request.messages)
        if not messages:
            raise HTTPException(status_code=400, detail="消息不能为空")
        if messages[-1].role != "user":
            raise HTTPException(status_code=400, detail="最后一条消息必须来自用户")

        try:
            # 只增强最后一条用户问题，保留原始对话历史，避免把检索上下文重复塞进多轮消息。
            enriched_messages = _attach_graph_context(
                messages,
                graph_reader=graph_reader,
                database_path=database_path,
                embedding_client=active_embedding_client,
            )
            reply = active_llm_client.generate_reply(enriched_messages)
        except (EmbeddingRequestError, EmbeddingResponseError) as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        except (LLMRequestError, LLMResponseError) as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

        return {
            "status": "success",
            "message": {"role": "assistant", "content": reply},
            "model": active_llm_client.config.model,
            "provider": active_llm_client.config.provider,
            "format": active_llm_client.config.format.value,
        }

    @app.get("/", response_model=None)
    def index() -> FileResponse | HTMLResponse:
        """返回已构建前端；未构建时提供最小占位入口。"""

        if static_index_path.exists():
            return FileResponse(static_index_path)

        return HTMLResponse(_placeholder_html())

    return app


class ChatMessagePayload(BaseModel):
    """前端传入的聊天消息。"""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    """聊天请求体。"""

    messages: list[ChatMessagePayload] = Field(min_length=1, max_length=50)


class RigelGraphReader:
    """读取 FalkorDBLite 中的 Rigel 图谱演示数据。"""

    def __init__(self, *, database_path: Path, graph_name: str) -> None:
        self._database_path = database_path
        self._graph_name = graph_name

    def summary(self) -> dict[str, object]:
        """统计默认可视化语义图谱的节点、边与节点类型分布。"""

        node_count = self._scalar_query(
            """
            MATCH (node:RigelNode)
            WHERE node.rigel_type IN $visible_node_types
            RETURN count(node)
            """,
            {"visible_node_types": list(VISIBLE_NODE_TYPES)},
        )
        edge_count = self._scalar_query(
            """
            MATCH (source:RigelNode)-[edge]->(target:RigelNode)
            WHERE source.rigel_type IN $visible_node_types
              AND target.rigel_type IN $visible_node_types
              AND type(edge) IN $visible_edge_types
            RETURN count(edge)
            """,
            {
                "visible_node_types": list(VISIBLE_NODE_TYPES),
                "visible_edge_types": list(VISIBLE_EDGE_TYPES),
            },
        )
        type_rows = self._query(
            """
            MATCH (node:RigelNode)
            WHERE node.rigel_type IN $visible_node_types
            RETURN node.rigel_type, count(node)
            ORDER BY count(node) DESC
            """,
            {"visible_node_types": list(VISIBLE_NODE_TYPES)},
        )
        return {
            "node_count": node_count,
            "edge_count": edge_count,
            "node_types": [
                {"type": node_type, "count": count}
                for node_type, count in type_rows
            ],
        }

    def graph(self, *, limit: int) -> dict[str, list[dict[str, object]]]:
        """读取默认可视化语义节点和这些节点之间的语义边。"""

        node_rows = self._query(
            """
            MATCH (node:RigelNode)
            WHERE node.rigel_type IN $visible_node_types
            RETURN node.id, properties(node)
            LIMIT $limit
            """,
            {"visible_node_types": list(VISIBLE_NODE_TYPES), "limit": limit},
        )
        nodes = [_format_node(node_id, properties) for node_id, properties in node_rows]
        node_ids = [str(node["id"]) for node in nodes]
        if not node_ids:
            return {"nodes": [], "edges": []}

        # 边只返回当前节点窗口内部的关系，避免前端收到指向缺失节点的悬空连线。
        edge_rows = self._query(
            """
            MATCH (source:RigelNode)-[edge]->(target:RigelNode)
            WHERE source.id IN $node_ids AND target.id IN $node_ids
              AND type(edge) IN $visible_edge_types
            RETURN source.id, target.id, type(edge), properties(edge)
            LIMIT $limit
            """,
            {
                "node_ids": node_ids,
                "visible_edge_types": list(VISIBLE_EDGE_TYPES),
                "limit": limit * 2,
            },
        )
        edges = [_format_edge(source_id, target_id, edge_type, properties) for source_id, target_id, edge_type, properties in edge_rows]
        return {"nodes": nodes, "edges": edges}

    def search(self, query: str, *, limit: int) -> list[dict[str, object]]:
        """在常见展示字段中做大小写不敏感搜索。"""

        normalized_query = query.lower()
        rows = self._query(
            """
            MATCH (node:RigelNode)
            WHERE node.rigel_type IN $visible_node_types
              AND (
                toLower(coalesce(node.display_name, '')) CONTAINS $query
                OR toLower(coalesce(node.qualified_name, '')) CONTAINS $query
                OR toLower(coalesce(node.relative_path, '')) CONTAINS $query
                OR toLower(coalesce(node.name, '')) CONTAINS $query
                OR toLower(coalesce(node.id, '')) CONTAINS $query
              )
            RETURN node.id, properties(node)
            LIMIT $limit
            """,
            {"visible_node_types": list(VISIBLE_NODE_TYPES), "query": normalized_query, "limit": limit},
        )
        return [_format_node(node_id, properties) for node_id, properties in rows]

    def recall(
        self,
        query_embedding: list[float],
        *,
        embedding_model: str,
        limit: int,
        expansion_limit: int,
    ) -> list[dict[str, object]]:
        """通过 Summary embedding 召回种子节点并补充一跳图谱上下文。"""

        rows = self._query(
            """
            MATCH (summary:RigelNode:Summary)-[:DESCRIBES]->(target:RigelNode)
            WHERE summary.purpose = $purpose
              AND summary.embedding_model = $embedding_model
              AND summary.embedding_dimensions = $embedding_dimensions
              AND target.rigel_type IN $visible_node_types
            RETURN summary.id, properties(summary), target.id, properties(target)
            LIMIT $scan_limit
            """,
            {
                "purpose": RETRIEVAL_SUMMARY_PURPOSE,
                "embedding_model": embedding_model,
                "embedding_dimensions": len(query_embedding),
                "visible_node_types": list(VISIBLE_NODE_TYPES),
                "scan_limit": DEFAULT_RECALL_SCAN_LIMIT,
            },
        )
        scored_results: list[dict[str, object]] = []
        for summary_id, summary_properties, target_id, target_properties in rows:
            embedding = _read_embedding(cast(Mapping[str, object], summary_properties))
            score = cosine_similarity(query_embedding, embedding)
            if score <= 0:
                continue
            node = _format_node(target_id, target_properties)
            scored_results.append(
                {
                    "score": score,
                    "summary": _format_summary(summary_id, summary_properties),
                    "node": node,
                    "related": self._related_nodes(str(target_id), limit=expansion_limit),
                }
            )

        scored_results.sort(
            key=lambda result: (
                -cast(float, result["score"]),
                str(cast(Mapping[str, object], result["node"])["label"]),
            )
        )
        return scored_results[:limit]

    def _related_nodes(self, node_id: str, *, limit: int) -> list[dict[str, object]]:
        rows = self._query(
            """
            MATCH (source:RigelNode)-[edge]->(target:RigelNode)
            WHERE (source.id = $node_id OR target.id = $node_id)
              AND source.rigel_type IN $visible_node_types
              AND target.rigel_type IN $visible_node_types
              AND type(edge) IN $visible_edge_types
            RETURN source.id, properties(source), target.id, properties(target), type(edge), properties(edge)
            LIMIT $limit
            """,
            {
                "node_id": node_id,
                "visible_node_types": list(VISIBLE_NODE_TYPES),
                "visible_edge_types": list(VISIBLE_EDGE_TYPES),
                "limit": limit,
            },
        )

        related_nodes: list[dict[str, object]] = []
        for source_id, source_properties, target_id, target_properties, edge_type, edge_properties in rows:
            source_node = _format_node(source_id, source_properties)
            target_node = _format_node(target_id, target_properties)
            related_node = target_node if str(source_id) == node_id else source_node
            related_nodes.append(
                {
                    "direction": "outgoing" if str(source_id) == node_id else "incoming",
                    "edge": _format_edge(source_id, target_id, edge_type, edge_properties),
                    "node": related_node,
                }
            )
        return related_nodes

    def _scalar_query(self, query: str, parameters: Mapping[str, object]) -> int:
        rows = self._query(query, parameters)
        if not rows:
            return 0
        return int(rows[0][0])

    def _query(self, query: str, parameters: Mapping[str, object]) -> list[list[Any]]:
        store = FalkorDBStore.connect(
            FalkorDBConfig(graph_name=self._graph_name, database_path=str(self._database_path))
        )
        result = store.graph.query(query, dict(parameters))
        return list(result.result_set)


def _read_graph_name(state_path: Path) -> str:
    """优先使用索引状态中的图名称。"""

    if not state_path.exists():
        return DEFAULT_GRAPH_NAME

    import json

    return cast(str, json.loads(state_path.read_text(encoding="utf-8"))["graph_name"])


def _ensure_database_exists(database_path: Path) -> None:
    """确保当前目录已存在 Rigel 本地图数据库。"""

    if not database_artifact_exists(database_path):
        raise HTTPException(
            status_code=404,
            detail=f"未找到数据库文件，请先在目标仓库执行 rigel index: {database_path}",
        )


def _resolve_llm_client(repository_path: Path, provided_client: RigelLLM | None) -> tuple[RigelLLM | None, str | None]:
    if provided_client is not None:
        # 测试和嵌入场景可以显式传入客户端，避免读取当前仓库的本地配置文件。
        return provided_client, None

    try:
        return RigelLLM(LLMConfig.from_repository(repository_path)), None
    except LLMConfigurationError as error:
        return None, str(error)


def _resolve_embedding_client(
    repository_path: Path,
    provided_client: RigelEmbedding | None,
) -> tuple[RigelEmbedding | None, str | None]:
    if provided_client is not None:
        # 测试和嵌入场景可以显式传入客户端，避免读取当前仓库的本地配置文件。
        return provided_client, None

    try:
        return RigelEmbedding(EmbeddingConfig.from_repository(repository_path)), None
    except EmbeddingConfigurationError as error:
        return None, str(error)


def _llm_status(llm_client: RigelLLM | None, configuration_error: str | None) -> dict[str, object]:
    if llm_client is None:
        return {
            "configured": False,
            "error": configuration_error,
        }

    config = llm_client.config
    return {
        "configured": True,
        "provider": config.provider,
        "format": config.format.value,
        "model": config.model,
        "base_url": config.base_url,
    }


def _embedding_status(embedding_client: RigelEmbedding | None, configuration_error: str | None) -> dict[str, object]:
    if embedding_client is None:
        return {
            "configured": False,
            "error": configuration_error,
        }

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


def _attach_graph_context(
    messages: list[LLMMessage],
    *,
    graph_reader: RigelGraphReader,
    database_path: Path,
    embedding_client: RigelEmbedding | None,
) -> list[LLMMessage]:
    if not database_artifact_exists(database_path):
        # 未初始化仓库仍允许纯 LLM 对话，避免 Web 页面因为缺少图数据库完全不可用。
        return messages
    if embedding_client is None:
        return messages

    context = _build_graph_context(messages[-1].content, graph_reader=graph_reader, embedding_client=embedding_client)
    if not context:
        return messages

    # 把检索结果写入用户问题前缀，让所有提供商都能通过普通文本获得同一份图谱上下文。
    contextualized_latest_message = LLMMessage(
        role="user",
        content=f"代码图谱搜索上下文：\n{context}\n\n用户问题：\n{messages[-1].content}",
    )
    return [*messages[:-1], contextualized_latest_message]


def _build_graph_context(
    query: str,
    *,
    graph_reader: RigelGraphReader,
    embedding_client: RigelEmbedding,
) -> str:
    query_embedding = embedding_client.embed_query(query)
    recall_results = graph_reader.recall(
        query_embedding,
        embedding_model=embedding_client.config.model,
        limit=DEFAULT_CHAT_CONTEXT_LIMIT,
        expansion_limit=DEFAULT_RECALL_EXPANSION_LIMIT,
    )
    if recall_results:
        return _format_recall_context(recall_results)

    fallback_nodes = graph_reader.search(query, limit=DEFAULT_CHAT_CONTEXT_LIMIT)
    if not fallback_nodes:
        return ""

    lines: list[str] = []
    for index, node in enumerate(fallback_nodes, start=1):
        properties = cast(Mapping[str, object], node["properties"])
        relative_path = _read_property(properties, "relative_path")
        qualified_name = _read_property(properties, "qualified_name")
        location = " / ".join(value for value in (qualified_name, relative_path) if value)
        suffix = f"：{location}" if location else ""
        lines.append(f"{index}. {node['label']}（{node['type']}，关键词回退）{suffix}")
    return "\n".join(lines)


def _format_recall_context(recall_results: list[dict[str, object]]) -> str:
    lines: list[str] = []
    for index, result in enumerate(recall_results, start=1):
        node = cast(Mapping[str, object], result["node"])
        summary = cast(Mapping[str, object], result["summary"])
        score = cast(float, result["score"])
        lines.append(
            f"{index}. {node['label']}（{node['type']}，向量分数 {score:.3f}）：{summary['text']}"
        )
        for related in cast(list[dict[str, object]], result["related"]):
            related_node = cast(Mapping[str, object], related["node"])
            edge = cast(Mapping[str, object], related["edge"])
            lines.append(
                f"   - {related['direction']} {edge['type']} {related_node['label']}（{related_node['type']}）"
            )
    return "\n".join(lines)


def _read_property(properties: Mapping[str, object], name: str) -> str:
    value = properties.get(name)
    return value if isinstance(value, str) else ""


def _format_node(node_id: str, properties: Mapping[str, object]) -> dict[str, object]:
    formatted_properties = dict(properties)
    return {
        "id": node_id,
        "type": str(formatted_properties["rigel_type"]),
        "label": _node_label(node_id, formatted_properties),
        "properties": formatted_properties,
    }


def _format_edge(
    source_id: str,
    target_id: str,
    edge_type: str,
    properties: Mapping[str, object],
) -> dict[str, object]:
    formatted_properties = dict(properties)
    return {
        "id": str(formatted_properties["id"]),
        "source": source_id,
        "target": target_id,
        "type": edge_type,
        "properties": formatted_properties,
    }


def _format_summary(summary_id: str, properties: Mapping[str, object]) -> dict[str, object]:
    return {
        "id": summary_id,
        "text": _read_property(properties, "text"),
        "embedding_model": _read_property(properties, "embedding_model"),
        "embedding_dimensions": _read_int_property(properties, "embedding_dimensions"),
        "source_hash": _read_property(properties, "source_hash"),
    }


def _read_int_property(properties: Mapping[str, object], name: str) -> int:
    value = properties.get(name)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _read_embedding(properties: Mapping[str, object]) -> list[float]:
    value = properties.get("embedding")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, list):
        return []

    embedding: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float):
            return []
        embedding.append(float(item))
    return embedding


def _node_label(node_id: str, properties: Mapping[str, object]) -> str:
    for key in ("display_name", "qualified_name", "relative_path", "name"):
        value = properties.get(key)
        if isinstance(value, str) and value:
            return value
    return node_id


def _placeholder_html() -> str:
    return """
    <!doctype html>
    <html lang="zh-CN">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Rigel Demo</title>
        <style>
          body { font-family: ui-sans-serif, system-ui, sans-serif; margin: 3rem; line-height: 1.6; }
          code { background: #f4f4f5; padding: 0.15rem 0.35rem; border-radius: 0.25rem; }
          a { color: #2563eb; }
        </style>
      </head>
      <body>
        <h1>Rigel Web Demo</h1>
        <p>前端暂未实现，后端 API 已启动。</p>
        <ul>
          <li><a href="/api/health"><code>/api/health</code></a></li>
          <li><a href="/api/summary"><code>/api/summary</code></a></li>
          <li><a href="/api/graph"><code>/api/graph</code></a></li>
        </ul>
      </body>
    </html>
    """
