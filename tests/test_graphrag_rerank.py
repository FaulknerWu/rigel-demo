from __future__ import annotations

from pathlib import Path
from typing import Any

from rigel_demo.config import GraphRAGConfig, LLMConfig, LLMConfigSection, RerankConfig
from rigel_demo.config.embedding import EmbeddingConfig, EmbeddingFormat, EmbeddingInputMode
from rigel_demo.graphrag.chat import LangGraphChatService
from rigel_demo.graphrag.tools import expand_graph_query
from rigel_demo.rerank import RerankResult


def test_vector_search_seeds_uses_multiple_queries_deduplicates_and_reranks() -> None:
    graph = FakeGraph(
        rows_by_embedding={
            (1.0, 0.0): [
                seed_row("summary-a", "node-a", "PaymentService", "付款服务摘要", 0.25),
                seed_row("summary-b", "node-b", "OrderRepository", "订单仓储摘要", 0.45),
            ],
            (0.0, 1.0): [
                seed_row("summary-a-better", "node-a", "PaymentService", "付款服务更相关摘要", 0.10),
                seed_row("summary-c", "node-c", "CheckoutController", "结账控制器摘要", 0.35),
            ],
        }
    )
    reranker = FakeReranker(
        [
            RerankResult(index=2, relevance_score=0.91),
            RerankResult(index=0, relevance_score=0.72),
            RerankResult(index=1, relevance_score=0.51),
        ]
    )
    service = LangGraphChatService(
        config=llm_config(),
        graphrag_config=GraphRAGConfig(host="127.0.0.1", port=6379, username=None, password=None),
        graph_name="rigel",
        embedding_client=FakeEmbedding(),
        reranker=reranker,
        chat_model=FakeChatModel(),
        graph=graph,
    )
    known_node_ids: set[str] = set()

    result = service._vector_search_seeds(
        {"query_texts": ["付款服务", "CheckoutController"]},
        fallback_query_text="付款流程在哪里处理？",
        known_node_ids=known_node_ids,
    )

    assert graph.vector_limits == [8, 8]
    assert reranker.query == "付款流程在哪里处理？"
    assert reranker.top_n == 3
    assert reranker.documents == ["付款服务更相关摘要", "订单仓储摘要", "结账控制器摘要"]
    assert known_node_ids == {"node-a", "node-b", "node-c"}
    assert [item["node"]["id"] for item in result["items"]] == ["node-c", "node-a", "node-b"]
    assert result["items"][0]["rerank_score"] == 0.91
    assert result["items"][1]["vector_score"] == 0.9


def test_graph_expansion_does_not_use_limit() -> None:
    graph = FakeGraph(rows_by_embedding={})
    service = LangGraphChatService(
        config=llm_config(),
        graphrag_config=GraphRAGConfig(host="127.0.0.1", port=6379, username=None, password=None),
        graph_name="rigel",
        embedding_client=FakeEmbedding(),
        reranker=FakeReranker([]),
        chat_model=FakeChatModel(),
        graph=graph,
    )
    known_node_ids = {"node-a"}

    service._expand_neighbors(
        {"node_ids": ["node-a"]},
        known_node_ids=known_node_ids,
        visited_node_ids=set(),
    )
    service._query_relation(
        {"node_ids": ["node-a"], "relation_type": "DEPENDS_ON"},
        known_node_ids=known_node_ids,
    )

    assert graph.relation_params == [
        {
            "node_id": "node-a",
            "visible_node_types": graph.relation_params[0]["visible_node_types"],
            "edge_types": graph.relation_params[0]["edge_types"],
        },
        {
            "node_id": "node-a",
            "visible_node_types": graph.relation_params[1]["visible_node_types"],
            "edge_types": ["DEPENDS_ON"],
        },
    ]
    assert "LIMIT" not in expand_graph_query("both")


def seed_row(summary_id: str, node_id: str, label: str, summary_text: str, distance: float) -> dict[str, object]:
    return {
        "summary_id": summary_id,
        "summary_properties": {
            "rigel_type": "Summary",
            "text": summary_text,
            "summary_model": "gpt-5.2",
            "embedding_model": "embedding-model",
            "embedding_dimensions": 2,
            "source_hash": "hash",
        },
        "node_id": node_id,
        "node_properties": {
            "rigel_type": "Entity",
            "display_name": label,
        },
        "distance": distance,
    }


def llm_config() -> LLMConfig:
    return LLMConfig(
        provider="openai",
        model="gpt-5.2",
        api_key="token",
        base_url=None,
        timeout_seconds=60,
        system_prompt="system",
        section=LLMConfigSection.CHAT,
    )


class FakeEmbedding:
    config = EmbeddingConfig(
        provider="openai",
        format=EmbeddingFormat.OPENAI_EMBEDDINGS,
        model="embedding-model",
        api_key="token",
        base_url=None,
        dimensions=2,
        timeout_seconds=60,
        batch_size=64,
        input_mode=EmbeddingInputMode.ARRAY,
    )

    def embed_query(self, text: str) -> list[float]:
        if text == "付款服务":
            return [1.0, 0.0]
        if text == "CheckoutController":
            return [0.0, 1.0]
        raise AssertionError(f"未知查询：{text}")


class FakeReranker:
    config = RerankConfig(
        provider="gitee_ai",
        base_url="https://ai.gitee.com/v1",
        model="Qwen3-Reranker-4B",
        api_key="token",
        timeout_seconds=60,
        top_n=5,
        candidate_limit_per_query=8,
    )

    def __init__(self, results: list[RerankResult]) -> None:
        self._results = results
        self.query = ""
        self.top_n = 0
        self.documents: list[str] = []

    def rerank(self, *, query: str, documents: list[str], top_n: int) -> list[RerankResult]:
        self.query = query
        self.top_n = top_n
        self.documents = documents
        return self._results[:top_n]


class FakeGraph:
    def __init__(self, rows_by_embedding: dict[tuple[float, ...], list[dict[str, object]]]) -> None:
        self._rows_by_embedding = rows_by_embedding
        self.vector_limits: list[int] = []
        self.relation_params: list[dict[str, object]] = []
        self.get_schema = ""

    def query(self, query: str, params: dict[str, object]) -> list[dict[str, object]]:
        if "query_embedding" not in params:
            self.relation_params.append(params)
            return []
        self.vector_limits.append(int(params["vector_limit"]))
        return self._rows_by_embedding[tuple(params["query_embedding"])]


class FakeChatModel:
    def bind_tools(self, tools: list[Any]) -> "FakeChatModel":
        return self
