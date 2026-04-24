"""Rigel Web 演示后端。"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from rigel_demo.cli import (
    DEFAULT_GRAPH_NAME,
    FALKORDB_DATABASE_FILE_NAME,
    RIGEL_WORKSPACE_DIRECTORY_NAME,
    WORKSPACE_STATE_FILE_NAME,
)
from rigel_demo.storage.falkordb_store import FalkorDBConfig, FalkorDBStore

DEFAULT_GRAPH_LIMIT = 500
DEFAULT_SEARCH_LIMIT = 50


def create_app(repository_path: Path | None = None) -> FastAPI:
    """创建基于当前仓库 `.rigel` 目录的 Web 演示应用。"""

    resolved_repository_path = (repository_path or Path.cwd()).resolve()
    workspace_path = resolved_repository_path / RIGEL_WORKSPACE_DIRECTORY_NAME
    database_path = workspace_path / FALKORDB_DATABASE_FILE_NAME
    state_path = workspace_path / WORKSPACE_STATE_FILE_NAME
    graph_name = _read_graph_name(state_path)
    graph_reader = RigelGraphReader(database_path=database_path, graph_name=graph_name)

    app = FastAPI(title="Rigel Demo", version="0.1.0")

    @app.get("/api/health")
    def health() -> dict[str, object]:
        """返回演示后端与本地数据库状态。"""

        return {
            "status": "ok" if database_path.exists() else "missing_database",
            "repository_path": str(resolved_repository_path),
            "workspace_path": str(workspace_path),
            "database_path": str(database_path),
            "graph_name": graph_name,
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

    @app.get("/")
    def index() -> HTMLResponse:
        """前端未实现前提供最小占位入口。"""

        return HTMLResponse(_placeholder_html())

    return app


class RigelGraphReader:
    """读取 FalkorDBLite 中的 Rigel 图谱演示数据。"""

    def __init__(self, *, database_path: Path, graph_name: str) -> None:
        self._database_path = database_path
        self._graph_name = graph_name

    def summary(self) -> dict[str, object]:
        """统计节点、边与节点类型分布。"""

        node_count = self._scalar_query("MATCH (node:RigelNode) RETURN count(node)")
        edge_count = self._scalar_query("MATCH (:RigelNode)-[edge]->(:RigelNode) RETURN count(edge)")
        type_rows = self._query(
            """
            MATCH (node:RigelNode)
            RETURN node.rigel_type, count(node)
            ORDER BY count(node) DESC
            """
        )
        return {
            "node_count": node_count,
            "edge_count": edge_count,
            "node_types": [
                {"type": node_type or "Unknown", "count": count}
                for node_type, count in type_rows
            ],
        }

    def graph(self, *, limit: int) -> dict[str, list[dict[str, object]]]:
        """读取一批节点和这些节点之间的边。"""

        node_rows = self._query(
            """
            MATCH (node:RigelNode)
            RETURN labels(node), node.id, properties(node)
            LIMIT $limit
            """,
            {"limit": limit},
        )
        nodes = [_format_node(labels, node_id, properties) for labels, node_id, properties in node_rows]
        node_ids = [str(node["id"]) for node in nodes]
        if not node_ids:
            return {"nodes": [], "edges": []}

        edge_rows = self._query(
            """
            MATCH (source:RigelNode)-[edge]->(target:RigelNode)
            WHERE source.id IN $node_ids AND target.id IN $node_ids
            RETURN source.id, target.id, type(edge), properties(edge)
            LIMIT $limit
            """,
            {"node_ids": node_ids, "limit": limit * 2},
        )
        edges = [_format_edge(source_id, target_id, edge_type, properties) for source_id, target_id, edge_type, properties in edge_rows]
        return {"nodes": nodes, "edges": edges}

    def search(self, query: str, *, limit: int) -> list[dict[str, object]]:
        """在常见展示字段中做大小写不敏感搜索。"""

        normalized_query = query.lower()
        rows = self._query(
            """
            MATCH (node:RigelNode)
            WHERE toLower(coalesce(node.display_name, '')) CONTAINS $query
               OR toLower(coalesce(node.qualified_name, '')) CONTAINS $query
               OR toLower(coalesce(node.relative_path, '')) CONTAINS $query
               OR toLower(coalesce(node.name, '')) CONTAINS $query
               OR toLower(coalesce(node.id, '')) CONTAINS $query
            RETURN labels(node), node.id, properties(node)
            LIMIT $limit
            """,
            {"query": normalized_query, "limit": limit},
        )
        return [_format_node(labels, node_id, properties) for labels, node_id, properties in rows]

    def _scalar_query(self, query: str) -> int:
        rows = self._query(query)
        if not rows:
            return 0
        return int(rows[0][0])

    def _query(self, query: str, parameters: Mapping[str, object] | None = None) -> list[list[Any]]:
        store = FalkorDBStore.connect(
            FalkorDBConfig(graph_name=self._graph_name, database_path=str(self._database_path))
        )
        result = store.graph.query(query, dict(parameters or {}))
        return list(result.result_set)


def _read_graph_name(state_path: Path) -> str:
    """优先使用初始化状态中的图名称。"""

    if not state_path.exists():
        return DEFAULT_GRAPH_NAME

    import json

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return DEFAULT_GRAPH_NAME

    graph_name = state.get("graph_name")
    return graph_name if isinstance(graph_name, str) and graph_name else DEFAULT_GRAPH_NAME


def _ensure_database_exists(database_path: Path) -> None:
    """确保当前目录已存在 Rigel 本地图数据库。"""

    if not database_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"未找到数据库文件，请先在目标仓库执行 rigel init: {database_path}",
        )


def _format_node(labels: list[str], node_id: str, properties: Mapping[str, object]) -> dict[str, object]:
    formatted_properties = dict(properties)
    node_type = str(formatted_properties.get("rigel_type") or _first_domain_label(labels))
    return {
        "id": node_id,
        "type": node_type,
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
        "id": str(formatted_properties.get("id") or f"{edge_type}:{source_id}:{target_id}"),
        "source": source_id,
        "target": target_id,
        "type": edge_type,
        "properties": formatted_properties,
    }


def _first_domain_label(labels: list[str]) -> str:
    for label in labels:
        if label != "RigelNode":
            return label
    return "Unknown"


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
