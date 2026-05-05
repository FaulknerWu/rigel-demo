"""GraphRAG 图谱工具的共享执行层。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from rigel_demo.config import GraphRAGConfig, RerankConfig
from rigel_demo.config.embedding import EmbeddingConfig
from rigel_demo.embedding import RigelEmbedding, build_rigel_embedding
from rigel_demo.graphrag.models import RigelGraphRAGError
from rigel_demo.graphrag.runtime import (
    EmbeddedFalkorDBRuntime,
    build_falkordb_graph,
    runtime_graphrag_config,
    start_embedded_falkordb_runtime,
)
from rigel_demo.graphrag.tools import (
    TOOL_EDGE_TYPES,
    direction_arg,
    edge_types_arg,
    expand_graph_query,
    filter_known_node_ids,
    filter_known_unvisited_node_ids,
    query_graph,
    relation_from_row,
    seed_row_values,
    string_arg,
    string_list_arg,
    tool_result,
)
from rigel_demo.project.summaries import RETRIEVAL_SUMMARY_PURPOSE
from rigel_demo.query.presentation import cosine_distance_to_similarity, format_node, format_summary
from rigel_demo.query.service import GraphExpansionDirection, VISIBLE_NODE_TYPES
from rigel_demo.rerank import RerankResult, RigelReranker, build_rigel_reranker


class RigelToolExecutor:
    """执行 GraphRAG 与 CLI 共用的受控图谱查询工具。"""

    def __init__(
        self,
        *,
        graph: Any,
        embedding_client: RigelEmbedding | None = None,
        reranker: RigelReranker | None = None,
    ) -> None:
        self._graph = graph
        self._embedding_client = embedding_client
        self._reranker = reranker

    def vector_search_seeds(
        self,
        args: Mapping[str, object],
        *,
        fallback_query_text: str = "",
        known_node_ids: set[str] | None = None,
    ) -> dict[str, object]:
        """按自然语言查询召回候选图谱节点。"""

        active_known_node_ids = known_node_ids if known_node_ids is not None else set()
        query_texts = string_list_arg(args, "query_texts")
        if not query_texts and fallback_query_text.strip():
            query_texts = [fallback_query_text.strip()]
        if not query_texts:
            return tool_result("vector_search_seeds", [], ["query_texts 不能为空"])
        if self._embedding_client is None or self._reranker is None:
            return tool_result("vector_search_seeds", [], ["vector_search_seeds 需要 embedding 与 rerank 配置"])

        candidates: dict[str, dict[str, object]] = {}
        for query_text in dict.fromkeys(query_texts):
            query_embedding = self._embedding_client.embed_query(query_text)
            rows = query_graph(
                self._graph,
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
                    "vector_limit": self._reranker.config.candidate_limit_per_query,
                },
            )
            for row in rows:
                summary_id, summary_properties, node_id, node_properties, distance = seed_row_values(row)
                vector_score = cosine_distance_to_similarity(float(distance))
                if vector_score <= 0:
                    continue
                node_key = str(node_id)
                current_candidate = candidates.get(node_key)
                if current_candidate is not None and float(current_candidate["vector_score"]) >= vector_score:
                    continue
                candidates[node_key] = {
                    "vector_score": vector_score,
                    "summary": format_summary(str(summary_id), cast(Mapping[str, object], summary_properties)),
                    "node": format_node(node_key, cast(Mapping[str, object], node_properties)),
                }

        candidate_items = list(candidates.values())
        if not candidate_items:
            return tool_result("vector_search_seeds", [], [])

        rerank_query = fallback_query_text.strip() or query_texts[0]
        rerank_top_n = min(self._reranker.config.top_n, len(candidate_items))
        rerank_results = self._reranker.rerank(
            query=rerank_query,
            documents=[summary_text(candidate["summary"]) for candidate in candidate_items],
            top_n=rerank_top_n,
        )
        items = self._reranked_seed_items(
            candidate_items,
            rerank_results,
            rerank_top_n=rerank_top_n,
            known_node_ids=active_known_node_ids,
        )
        return tool_result("vector_search_seeds", items[: self._reranker.config.top_n], [])

    def expand_neighbors(
        self,
        args: Mapping[str, object],
        *,
        known_node_ids: set[str] | None = None,
        visited_node_ids: set[str] | None = None,
        enforce_known_nodes: bool = True,
        enforce_unvisited_nodes: bool = True,
    ) -> dict[str, object]:
        """按节点 ID 批量读取一跳邻接关系。"""

        requested_node_ids = string_list_arg(args, "node_ids")
        direction = direction_arg(args)
        edge_types, edge_warnings = edge_types_arg(args.get("edge_types"), default=list(TOOL_EDGE_TYPES))
        warnings = [*edge_warnings]
        active_known_node_ids = known_node_ids if known_node_ids is not None else set(requested_node_ids)
        active_visited_node_ids = visited_node_ids if visited_node_ids is not None else set()

        if enforce_known_nodes and enforce_unvisited_nodes:
            node_ids = filter_known_unvisited_node_ids(
                requested_node_ids,
                known_node_ids=active_known_node_ids,
                visited_node_ids=active_visited_node_ids,
                warnings=warnings,
            )
        elif enforce_known_nodes:
            node_ids = filter_known_node_ids(
                requested_node_ids,
                known_node_ids=active_known_node_ids,
                warnings=warnings,
            )
        else:
            node_ids = list(dict.fromkeys(requested_node_ids))

        relations = self.relations_for_node_ids(
            node_ids,
            direction=direction,
            edge_types=edge_types,
            known_node_ids=active_known_node_ids,
        )
        active_visited_node_ids.update(node_ids)
        return tool_result("expand_neighbors", relations, warnings)

    def query_relation(
        self,
        args: Mapping[str, object],
        *,
        known_node_ids: set[str] | None = None,
        enforce_known_nodes: bool = True,
    ) -> dict[str, object]:
        """按单一关系类型精确查询一跳关系。"""

        requested_node_ids = string_list_arg(args, "node_ids")
        relation_type = string_arg(args, "relation_type", default="")
        direction = direction_arg(args)
        warnings: list[str] = []
        if relation_type not in TOOL_EDGE_TYPES:
            return tool_result("query_relation", [], [f"非法关系类型已忽略：{relation_type}"])

        active_known_node_ids = known_node_ids if known_node_ids is not None else set(requested_node_ids)
        if enforce_known_nodes:
            node_ids = filter_known_node_ids(
                requested_node_ids,
                known_node_ids=active_known_node_ids,
                warnings=warnings,
            )
        else:
            node_ids = list(dict.fromkeys(requested_node_ids))

        relations = self.relations_for_node_ids(
            node_ids,
            direction=direction,
            edge_types=[relation_type],
            known_node_ids=active_known_node_ids,
        )
        return tool_result("query_relation", relations, warnings)

    def relations_for_node_ids(
        self,
        node_ids: list[str],
        *,
        direction: GraphExpansionDirection,
        edge_types: list[str],
        known_node_ids: set[str],
    ) -> list[dict[str, object]]:
        relations: list[dict[str, object]] = []
        for node_id in node_ids:
            rows = query_graph(
                self._graph,
                expand_graph_query(direction),
                {
                    "node_id": node_id,
                    "visible_node_types": list(VISIBLE_NODE_TYPES),
                    "edge_types": edge_types,
                },
            )
            for row in rows:
                relation = relation_from_row(row, origin_node_id=node_id)
                known_node_ids.add(str(cast(Mapping[str, object], relation["node"])["id"]))
                relations.append(relation)
        return relations

    def _reranked_seed_items(
        self,
        candidate_items: list[dict[str, object]],
        rerank_results: Sequence[object],
        *,
        rerank_top_n: int,
        known_node_ids: set[str],
    ) -> list[dict[str, object]]:
        items: list[dict[str, object]] = []
        used_indexes: set[int] = set()
        for rerank_result in normalized_rerank_results(rerank_results):
            if rerank_result.index < 0 or rerank_result.index >= len(candidate_items):
                continue
            candidate = candidate_items[rerank_result.index]
            node = cast(Mapping[str, object], candidate["node"])
            known_node_ids.add(str(node["id"]))
            used_indexes.add(rerank_result.index)
            items.append(
                {
                    "tool": "vector_search_seeds",
                    "score": rerank_result.relevance_score,
                    "vector_score": candidate["vector_score"],
                    "rerank_score": rerank_result.relevance_score,
                    "summary": candidate["summary"],
                    "node": candidate["node"],
                }
            )

        if len(items) < rerank_top_n:
            fallback_items = [
                (index, candidate)
                for index, candidate in enumerate(candidate_items)
                if index not in used_indexes
            ]
            fallback_items.sort(
                key=lambda item: (
                    -float(item[1]["vector_score"]),
                    str(cast(Mapping[str, object], item[1]["node"])["label"]),
                )
            )
            for _, candidate in fallback_items[: rerank_top_n - len(items)]:
                node = cast(Mapping[str, object], candidate["node"])
                known_node_ids.add(str(node["id"]))
                items.append(
                    {
                        "tool": "vector_search_seeds",
                        "score": candidate["vector_score"],
                        "vector_score": candidate["vector_score"],
                        "rerank_score": None,
                        "summary": candidate["summary"],
                        "node": candidate["node"],
                    }
                )

        items.sort(
            key=lambda item: (
                -(float(item["rerank_score"]) if isinstance(item["rerank_score"], int | float) else -1.0),
                -float(item["vector_score"]),
                str(cast(Mapping[str, object], item["node"])["label"]),
            )
        )
        return items


class RepositoryToolExecutor:
    """管理本地仓库图数据库运行时并暴露工具执行器。"""

    def __init__(self, *, executor: RigelToolExecutor, embedded_database: EmbeddedFalkorDBRuntime | None) -> None:
        self.executor = executor
        self._embedded_database = embedded_database

    def close(self) -> None:
        if self._embedded_database is not None:
            self._embedded_database.client.close()

    def __enter__(self) -> "RepositoryToolExecutor":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()


def build_repository_tool_executor(
    *,
    repository_path: Path,
    graph_name: str,
    database_path: Path | None = None,
    embedding_client: RigelEmbedding | None = None,
) -> RepositoryToolExecutor:
    """按仓库配置构建 CLI 可直接使用的图谱工具执行器。"""

    graphrag_config = GraphRAGConfig.from_repository(repository_path)
    rerank_config = RerankConfig.from_repository(repository_path)
    active_embedding_client = embedding_client or build_rigel_embedding(EmbeddingConfig.from_repository(repository_path))
    embedded_database = (
        start_embedded_falkordb_runtime(database_path=database_path, host=graphrag_config.host)
        if database_path is not None
        else None
    )
    graph = build_falkordb_graph(
        config=runtime_graphrag_config(graphrag_config, embedded_database),
        graph_name=graph_name,
    )
    return RepositoryToolExecutor(
        executor=RigelToolExecutor(
            graph=graph,
            embedding_client=active_embedding_client,
            reranker=build_rigel_reranker(rerank_config),
        ),
        embedded_database=embedded_database,
    )


def build_repository_graph_executor(
    *,
    repository_path: Path,
    graph_name: str,
    database_path: Path | None = None,
) -> RepositoryToolExecutor:
    """按仓库配置构建仅用于图查询的运行时。"""

    graphrag_config = GraphRAGConfig.from_repository(repository_path)
    embedded_database = (
        start_embedded_falkordb_runtime(database_path=database_path, host=graphrag_config.host)
        if database_path is not None
        else None
    )
    graph = build_falkordb_graph(
        config=runtime_graphrag_config(graphrag_config, embedded_database),
        graph_name=graph_name,
    )
    return RepositoryToolExecutor(
        executor=RigelToolExecutor(graph=graph),
        embedded_database=embedded_database,
    )


def summary_text(summary: object) -> str:
    if isinstance(summary, Mapping):
        text = summary.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
    raise RigelGraphRAGError("Rerank 候选 Summary 缺少 text")


def normalized_rerank_results(results: Sequence[object]) -> list[RerankResult]:
    normalized_results: list[RerankResult] = []
    for result in results:
        if isinstance(result, RerankResult):
            normalized_results.append(result)
            continue
        if isinstance(result, Mapping):
            index = result.get("index")
            relevance_score = result.get("relevance_score")
            if isinstance(index, int) and not isinstance(index, bool) and isinstance(relevance_score, int | float):
                normalized_results.append(RerankResult(index=index, relevance_score=float(relevance_score)))
    return normalized_results
