from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from fastapi.testclient import TestClient

from rigel_demo.core import EdgeType, Entity, GraphEdge, GraphIR, Module, Repository
from rigel_demo.embedding import EmbeddingConfig, EmbeddingFormat
from rigel_demo.indexing.retrieval_summaries import attach_retrieval_summaries
from rigel_demo.llm import LLMConfig, LLMConfigSection, LLMMessage
from rigel_demo.storage import FalkorDBConfig, FalkorDBStore
from rigel_demo.web.app import create_app


class WebAppLLMTest(TestCase):
    def test_chat_returns_configuration_error_when_llm_is_missing(self) -> None:
        with TemporaryDirectory() as workspace:
            client = TestClient(create_app(Path(workspace)))

            response = client.post("/api/chat", json={"messages": [{"role": "user", "content": "你好"}]})

        self.assertEqual(response.status_code, 503)
        self.assertIn(".rigel/config.json", response.json()["detail"])

    def test_chat_uses_injected_llm_client(self) -> None:
        fake_llm = _FakeRigelLLM()
        with TemporaryDirectory() as workspace:
            client = TestClient(create_app(Path(workspace), chat_client=fake_llm))

            response = client.post("/api/chat", json={"messages": [{"role": "user", "content": "你好"}]})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["message"], {"role": "assistant", "content": "测试回复"})
        self.assertEqual(fake_llm.messages, [LLMMessage(role="user", content="你好")])

    def test_recall_returns_summary_seed_and_related_context(self) -> None:
        fake_embedding = _FakeEmbeddingClient()
        with TemporaryDirectory() as workspace:
            repository_path = Path(workspace)
            _write_demo_graph(repository_path, fake_embedding)
            client = TestClient(create_app(repository_path, embedding_client=fake_embedding))

            response = client.get("/api/recall", params={"q": "PaymentService", "limit": "3"})

        self.assertEqual(response.status_code, 200)
        result = response.json()["results"][0]
        self.assertGreater(result["score"], 0)
        self.assertEqual(result["node"]["label"], "PaymentService")
        self.assertIn("PaymentService", result["summary"]["text"])
        self.assertEqual(result["related"][0]["edge"]["type"], "CONTAINS")

    def test_chat_attaches_vector_recall_context_when_graph_exists(self) -> None:
        fake_llm = _FakeRigelLLM()
        fake_embedding = _FakeEmbeddingClient()
        with TemporaryDirectory() as workspace:
            repository_path = Path(workspace)
            _write_demo_graph(repository_path, fake_embedding)
            client = TestClient(create_app(repository_path, chat_client=fake_llm, embedding_client=fake_embedding))

            response = client.post("/api/chat", json={"messages": [{"role": "user", "content": "PaymentService 做什么"}]})

        self.assertEqual(response.status_code, 200)
        self.assertIn("代码图谱搜索上下文", fake_llm.messages[0].content)
        self.assertIn("向量分数", fake_llm.messages[0].content)
        self.assertIn("PaymentService", fake_llm.messages[0].content)


def _write_demo_graph(repository_path: Path, embedding_client: "_FakeEmbeddingClient") -> None:
    workspace_path = repository_path / ".rigel"
    database_path = workspace_path / "falkordb.db"
    graph = GraphIR()
    repository = Repository(repo_id="repo:demo", name="demo")
    module = Module(
        module_id="module:demo:root",
        name="root",
        root_path=".",
        ecosystem="maven",
        zone="prod",
    )
    entity = Entity(
        entity_id="entity:demo:PaymentService",
        entity_key="java:demo.PaymentService",
        display_name="PaymentService",
        qualified_name="demo.PaymentService",
        kind_norm="class",
        kind_raw="class_declaration",
        origin="internal",
        semantic_hash="sha256:payment-service",
    )
    graph.add_node(repository)
    graph.add_node(module)
    graph.add_node(entity)
    graph.add_edge(GraphEdge.create(EdgeType.CONTAINS, repository.repo_id, module.module_id, kind="physical-membership"))
    graph.add_edge(GraphEdge.create(EdgeType.CONTAINS, module.module_id, entity.entity_id, kind="physical-membership"))
    attach_retrieval_summaries(graph, embedding_client=embedding_client, summary_client=_FakeSummaryClient())
    FalkorDBStore.connect(FalkorDBConfig(graph_name="rigel", database_path=str(database_path))).upsert_graph(graph)


class _FakeRigelLLM:
    def __init__(self) -> None:
        self.config = LLMConfig(
            provider="openai",
            model="fake-model",
            api_key="fake-key",
            base_url=None,
            timeout_seconds=1,
            system_prompt="系统提示",
            section=LLMConfigSection.CHAT,
        )
        self.messages: list[LLMMessage] = []

    def generate_reply(self, messages: list[LLMMessage]) -> str:
        self.messages = messages
        return "测试回复"


class _FakeEmbeddingClient:
    def __init__(self) -> None:
        self.config = EmbeddingConfig(
            provider="openai",
            format=EmbeddingFormat.OPENAI_EMBEDDINGS,
            model="text-embedding-3-small",
            api_key="fake-key",
            base_url=None,
            dimensions=3,
            timeout_seconds=1,
            batch_size=8,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embedding_for_text(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embedding_for_text(text)

    def _embedding_for_text(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0] if "PaymentService" in text else [0.0, 1.0, 0.0]


class _FakeSummaryClient:
    def __init__(self) -> None:
        self.config = LLMConfig(
            provider="openai",
            model="summary-model",
            api_key="fake-key",
            base_url=None,
            timeout_seconds=1,
            system_prompt="摘要提示",
            section=LLMConfigSection.SUMMARY,
        )

    def generate_reply(self, messages: list[LLMMessage]) -> str:
        return "PaymentService 处理付款流程"
