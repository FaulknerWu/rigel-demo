from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from fastapi.testclient import TestClient

from rigel_demo.core import Anchor, EdgeType, Entity, File, GraphEdge, GraphIR, Module, Repository
from rigel_demo.embedding import EmbeddingConfig, EmbeddingConfigurationError, EmbeddingFormat
from rigel_demo.indexing.retrieval_summaries import attach_retrieval_summaries
from rigel_demo.llm import (
    DEFAULT_CHAT_SYSTEM_PROMPT,
    DEFAULT_SUMMARY_SYSTEM_PROMPT,
    LLMConfig,
    LLMConfigSection,
    LLMConfigurationError,
    LLMMessage,
)
from rigel_demo.agent.langgraph_agent import AgentReply, ToolCallTrace
from rigel_demo.agent.service import RepositorySourceReader, SourceLineRangeError
from rigel_demo.storage import FalkorDBConfig, FalkorDBStore
from rigel_demo.web.app import create_app


class WebAppLLMTest(TestCase):
    def test_create_app_requires_chat_config_at_startup(self) -> None:
        fake_embedding = _FakeEmbeddingClient()
        with TemporaryDirectory() as workspace:
            repository_path = Path(workspace)
            _write_demo_graph(repository_path, fake_embedding, write_config=False)

            with self.assertRaisesRegex(LLMConfigurationError, ".rigel/config.json"):
                create_app(repository_path, embedding_client=fake_embedding)

    def test_create_app_requires_embedding_config_at_startup(self) -> None:
        with TemporaryDirectory() as workspace:
            repository_path = Path(workspace)
            _write_demo_graph(repository_path, _FakeEmbeddingClient(), write_config=False)
            repository_path.joinpath(".rigel", "config.json").write_text(
                json.dumps(
                    {
                        "chat": _demo_llm_config(DEFAULT_CHAT_SYSTEM_PROMPT),
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(EmbeddingConfigurationError, "embedding"):
                create_app(repository_path, chat_client=_FakeGraphAgent())

    def test_chat_uses_injected_agent_client(self) -> None:
        fake_agent = _FakeGraphAgent()
        fake_embedding = _FakeEmbeddingClient()
        with TemporaryDirectory() as workspace:
            repository_path = Path(workspace)
            _write_demo_graph(repository_path, fake_embedding)
            client = TestClient(create_app(repository_path, chat_client=fake_agent, embedding_client=fake_embedding))

            response = client.post("/api/chat", json={"messages": [{"role": "user", "content": "PaymentService 做什么"}]})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["message"], {"role": "assistant", "content": "测试回复"})
        self.assertEqual(payload["tool_calls"], [{"name": "recall", "args": {"query": "PaymentService 做什么"}}])
        self.assertEqual(fake_agent.messages[0], LLMMessage(role="user", content="PaymentService 做什么"))

    def test_recall_uses_falkordb_vector_index(self) -> None:
        fake_embedding = _FakeEmbeddingClient()
        with TemporaryDirectory() as workspace:
            repository_path = Path(workspace)
            database_path = _write_demo_graph(repository_path, fake_embedding)
            store = FalkorDBStore.connect(FalkorDBConfig(graph_name="rigel", database_path=str(database_path)))

            indexes = store.graph.query("CALL db.indexes()").result_set
            vector_rows = store.graph.query(
                "CALL db.idx.vector.queryNodes('Summary', 'embedding', 3, vecf32([1.0, 0.0, 0.0])) "
                "YIELD node, score RETURN node.id, score"
            ).result_set

        summary_index = next(index for index in indexes if index[0] == "Summary")
        self.assertEqual(summary_index[2]["embedding"], ["VECTOR"])
        self.assertEqual(summary_index[3]["embedding"]["dimension"], 3)
        self.assertEqual(summary_index[3]["embedding"]["similarityFunction"], "cosine")
        self.assertTrue(any(row[0].startswith("summary:") for row in vector_rows))

    def test_source_reader_rejects_non_integer_line_ranges(self) -> None:
        with TemporaryDirectory() as workspace:
            repository_path = Path(workspace)
            repository_path.joinpath("Demo.java").write_text("class Demo {}\n", encoding="utf-8")
            source_reader = RepositorySourceReader(repository_path)

            invalid_cases = [
                {"start_line": "1", "end_line": 1, "max_lines": 1},
                {"start_line": 1, "end_line": 1.0, "max_lines": 1},
                {"start_line": 1, "end_line": 1, "max_lines": None},
                {"start_line": True, "end_line": 1, "max_lines": 1},
            ]

            for invalid_case in invalid_cases:
                with self.subTest(invalid_case=invalid_case):
                    with self.assertRaisesRegex(SourceLineRangeError, "必须大于 0"):
                        source_reader.read_slice("Demo.java", **invalid_case)

    def test_chat_returns_agent_tool_trace_when_graph_exists(self) -> None:
        fake_agent = _FakeGraphAgent(
            AgentReply(
                content="PaymentService 处理付款流程",
                tool_calls=[
                    ToolCallTrace(name="recall", args={"query": "PaymentService 做什么"}),
                    ToolCallTrace(
                        name="source",
                        args={
                            "path": "src/main/java/demo/PaymentService.java",
                            "start_line": 3,
                            "end_line": 5,
                        },
                    ),
                ],
            )
        )
        fake_embedding = _FakeEmbeddingClient()
        with TemporaryDirectory() as workspace:
            repository_path = Path(workspace)
            _write_demo_graph(repository_path, fake_embedding)
            client = TestClient(create_app(repository_path, chat_client=fake_agent, embedding_client=fake_embedding))

            response = client.post("/api/chat", json={"messages": [{"role": "user", "content": "PaymentService 做什么"}]})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["message"]["content"], "PaymentService 处理付款流程")
        self.assertEqual([tool_call["name"] for tool_call in payload["tool_calls"]], ["recall", "source"])

    def test_chat_keeps_raw_messages_for_agent(self) -> None:
        fake_agent = _FakeGraphAgent(AgentReply(content="没有找到确定证据", tool_calls=[]))
        fake_embedding = _FakeEmbeddingClient()
        with TemporaryDirectory() as workspace:
            repository_path = Path(workspace)
            _write_demo_graph(repository_path, fake_embedding)
            client = TestClient(create_app(repository_path, chat_client=fake_agent, embedding_client=fake_embedding))

            response = client.post("/api/chat", json={"messages": [{"role": "user", "content": "NoMatch"}]})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["tool_calls"], [])
        self.assertEqual(fake_agent.messages[0], LLMMessage(role="user", content="NoMatch"))

    def test_incremental_index_endpoint_returns_shared_result_payload(self) -> None:
        fake_embedding = _FakeEmbeddingClient()
        with TemporaryDirectory() as workspace:
            repository_path = Path(workspace)
            _write_demo_graph(repository_path, fake_embedding)
            client = TestClient(create_app(repository_path, embedding_client=fake_embedding))

            with patch("rigel_demo.web.app.index_repository_workspace") as index_repository_workspace:
                index_repository_workspace.return_value = SimpleNamespace(
                    index_mode="incremental",
                    added_files=("src/main/java/demo/Added.java",),
                    modified_files=(),
                    deleted_files=(),
                    skipped_file_count=1,
                    indexed_file_count=1,
                    deleted_node_count=0,
                    graph_node_count=12,
                    graph_edge_count=9,
                    duration_ms=25,
                )

                response = client.post("/api/index/incremental")

        self.assertEqual(response.status_code, 200)
        payload = response.json()["result"]
        self.assertEqual(payload["mode"], "incremental")
        self.assertEqual(payload["added_files"], ["src/main/java/demo/Added.java"])
        self.assertEqual(payload["skipped_file_count"], 1)
        self.assertEqual(payload["graph_node_count"], 12)
        index_repository_workspace.assert_called_once_with(repository_path.resolve(), incremental=True)

    def test_store_delete_file_subgraphs_removes_file_owned_metadata(self) -> None:
        fake_embedding = _FakeEmbeddingClient()
        with TemporaryDirectory() as workspace:
            repository_path = Path(workspace)
            database_path = _write_demo_graph(repository_path, fake_embedding)
            store = FalkorDBStore.connect(FalkorDBConfig(graph_name="rigel", database_path=str(database_path)))

            deleted_count = store.delete_file_subgraphs(["src/main/java/demo/PaymentService.java"])
            remaining_files = store.graph.query("MATCH (file:RigelNode:File) RETURN count(file)").result_set
            remaining_summaries = store.graph.query("MATCH (summary:RigelNode:Summary) RETURN count(summary)").result_set
            remaining_repositories = store.graph.query("MATCH (repo:RigelNode:Repository) RETURN count(repo)").result_set

        self.assertGreater(deleted_count, 0)
        self.assertEqual(remaining_files[0][0], 0)
        self.assertEqual(remaining_summaries[0][0], 1)
        self.assertEqual(remaining_repositories[0][0], 1)


def _write_demo_graph(
    repository_path: Path,
    embedding_client: "_FakeEmbeddingClient",
    *,
    write_config: bool = True,
) -> Path:
    workspace_path = repository_path / ".rigel"
    database_path = workspace_path / "falkordb.db"
    static_path = workspace_path / "web" / "static"
    source_path = repository_path / "src" / "main" / "java" / "demo" / "PaymentService.java"
    workspace_path.mkdir(parents=True, exist_ok=True)
    static_path.joinpath("assets").mkdir(parents=True, exist_ok=True)
    static_path.joinpath("index.html").write_text("<!doctype html><div id=\"root\"></div>\n", encoding="utf-8")
    workspace_path.joinpath("rigel.json").write_text('{"graph_name": "rigel"}\n', encoding="utf-8")
    if write_config:
        workspace_path.joinpath("config.json").write_text(_demo_config_json(), encoding="utf-8")
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        "\n".join(
            [
                "package demo;",
                "",
                "public class PaymentService {",
                "    public String pay() {",
                "        return \"paid\";",
                "    }",
                "}",
                "",
                "",
                "",
                "",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    graph = GraphIR()
    repository = Repository(repo_id="repo:demo", name="demo")
    module = Module(
        module_id="module:demo:root",
        name="root",
        root_path=".",
        ecosystem="maven",
        zone="prod",
    )
    file = File(
        file_id="file:demo:src/main/java/demo/PaymentService.java",
        relative_path="src/main/java/demo/PaymentService.java",
        language="java",
        zone="prod",
        content_hash="sha256:payment-file",
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
    definition_anchor = Anchor(
        anchor_id="anchor:entity:demo:PaymentService:definition",
        start_line=3,
        start_col=1,
        end_line=12,
        end_col=2,
        role="definition",
    )
    body_anchor = Anchor(
        anchor_id="anchor:entity:demo:PaymentService:body",
        start_line=3,
        start_col=29,
        end_line=12,
        end_col=2,
        role="body",
    )
    graph.add_node(repository)
    graph.add_node(module)
    graph.add_node(file)
    graph.add_node(entity)
    graph.add_node(definition_anchor)
    graph.add_node(body_anchor)
    graph.add_edge(GraphEdge.create(EdgeType.CONTAINS, repository.repo_id, module.module_id, kind="physical-membership"))
    graph.add_edge(GraphEdge.create(EdgeType.CONTAINS, module.module_id, file.file_id, kind="physical-membership"))
    graph.add_edge(GraphEdge.create(EdgeType.CONTAINS, file.file_id, entity.entity_id, kind="physical-membership"))
    graph.add_edge(GraphEdge.create(EdgeType.HAS_ANCHOR, entity.entity_id, definition_anchor.anchor_id, role="definition"))
    graph.add_edge(GraphEdge.create(EdgeType.HAS_ANCHOR, entity.entity_id, body_anchor.anchor_id, role="body"))
    attach_retrieval_summaries(graph, embedding_client=embedding_client, summary_client=_FakeSummaryClient())
    FalkorDBStore.connect(FalkorDBConfig(graph_name="rigel", database_path=str(database_path))).upsert_graph(graph)
    return database_path


def _demo_config_json() -> str:
    return json.dumps(
        {
            "chat": _demo_llm_config(DEFAULT_CHAT_SYSTEM_PROMPT),
            "summary": {
                **_demo_llm_config(DEFAULT_SUMMARY_SYSTEM_PROMPT),
                "temperature": 0,
                "max_output_tokens": 300,
            },
            "embedding": {
                "provider": "openai",
                "format": "openai_embeddings",
                "model": "text-embedding-3-small",
                "api_key": "fake-embedding-key",
                "base_url": None,
                "dimensions": 3,
                "timeout_seconds": 60,
                "batch_size": 8,
            },
        },
        ensure_ascii=False,
        indent=2,
    ) + "\n"


def _demo_llm_config(system_prompt: str) -> dict[str, object]:
    return {
        "provider": "openai",
        "model": "gpt-5.2",
        "api_key": "fake-key",
        "base_url": None,
        "timeout_seconds": 60,
        "temperature": None,
        "max_output_tokens": None,
        "system_prompt": system_prompt,
    }


class _FakeGraphAgent:
    def __init__(self, reply: AgentReply | None = None) -> None:
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
        self.reply = reply or AgentReply(
            content="测试回复",
            tool_calls=[ToolCallTrace(name="recall", args={"query": "PaymentService 做什么"})],
        )

    def generate_reply(self, messages: list[LLMMessage]) -> AgentReply:
        self.messages = messages
        return self.reply


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
