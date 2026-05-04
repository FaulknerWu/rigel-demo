import hashlib
from pathlib import Path
from threading import Lock
from tempfile import TemporaryDirectory
from time import sleep
from unittest import TestCase
from unittest.mock import patch

from rigel_demo.entities import Entity, Module, Summary
from rigel_demo.graph import EdgeType, GraphEdge, NodeType
from rigel_demo.graph.ir import GraphIR, GraphNode
from rigel_demo.embedding import EmbeddingConfig, EmbeddingFormat, EmbeddingInputMode
from rigel_demo.project import repository_indexer
from rigel_demo.project.summaries import attach_retrieval_summaries
from rigel_demo.java.requests import DEFAULT_LSP_TIMEOUT_SECONDS, GENERATED_ZONE
from rigel_demo.llm import LLMConfig, LLMConfigSection, LLMMessage


class RepositoryIndexerTest(TestCase):
    def test_index_repository_reports_file_progress(self) -> None:
        with TemporaryDirectory() as workspace:
            repository_path = Path(workspace)
            first_path = repository_path / "src" / "main" / "java" / "demo" / "First.java"
            second_path = repository_path / "src" / "main" / "java" / "demo" / "Second.java"
            first_path.parent.mkdir(parents=True)
            first_path.write_text("package demo; public class First {}\n", encoding="utf-8")
            second_path.write_text("package demo; public class Second {}\n", encoding="utf-8")
            progress_events: list[tuple[str, int, int, str]] = []

            with patch.object(repository_indexer, "enrich_java_semantic_edges") as enrich_java_semantic_edges:
                enrich_java_semantic_edges.side_effect = lambda graph, *, request: graph

                repository_indexer.index_repository(
                    repository_path,
                    embedding_client=_FakeEmbeddingClient(),
                    summary_client=_FakeSummaryClient(),
                    progress_reporter=lambda progress: progress_events.append(
                        (progress.stage, progress.current, progress.total, progress.detail)
                    ),
                )

        java_parse_events = [event for event in progress_events if event[0] == "java_parse"]
        self.assertEqual(
            java_parse_events,
            [
                ("java_parse", 1, 2, "src/main/java/demo/First.java"),
                ("java_parse", 2, 2, "src/main/java/demo/Second.java"),
            ],
        )
        self.assertIn("semantic_edges", {event[0] for event in progress_events})
        self.assertIn("summary_text", {event[0] for event in progress_events})
        self.assertIn("summary_embedding", {event[0] for event in progress_events})

    def test_index_repository_uses_real_lsp_client_by_default(self) -> None:
        with TemporaryDirectory() as workspace:
            repository_path = Path(workspace)
            source_path = repository_path / "src" / "main" / "java" / "demo" / "App.java"
            source_path.parent.mkdir(parents=True)
            source_path.write_text(
                "package demo; public class App { public String run() { return \"ok\"; } }\n",
                encoding="utf-8",
            )

            with patch.object(repository_indexer, "enrich_java_semantic_edges") as enrich_java_semantic_edges:
                enrich_java_semantic_edges.side_effect = lambda graph, *, request: graph

                result = repository_indexer.index_repository(
                    repository_path,
                    embedding_client=_FakeEmbeddingClient(),
                    summary_client=_FakeSummaryClient(),
                )

        self.assertEqual(result.indexed_file_count, 1)
        self.assertEqual(enrich_java_semantic_edges.call_count, 1)
        self.assertNotIn("lsp_client", enrich_java_semantic_edges.call_args.kwargs)
        self.assertEqual(
            enrich_java_semantic_edges.call_args.kwargs["request"].lsp_timeout_seconds,
            DEFAULT_LSP_TIMEOUT_SECONDS,
        )
        self.assertTrue(any(node.type == NodeType.SUMMARY for node in result.graph.nodes))
        self.assertTrue(any(edge.type == EdgeType.DESCRIBES for edge in result.graph.edges))
        summary_nodes = [node for node in result.graph.nodes if node.type == NodeType.SUMMARY]
        self.assertTrue(any(node.properties["summary_model"] == "summary-model" for node in summary_nodes))
        self.assertTrue(any("模型摘要" in str(node.properties["text"]) for node in summary_nodes))

    def test_index_repository_assigns_module_and_zone_per_java_source_root(self) -> None:
        with TemporaryDirectory() as workspace:
            repository_path = Path(workspace)
            (repository_path / "services" / "billing").mkdir(parents=True)
            (repository_path / "services" / "billing" / "pom.xml").write_text("<project />\n", encoding="utf-8")
            production_path = repository_path / "services" / "billing" / "src" / "main" / "java" / "demo" / "PaymentService.java"
            test_path = repository_path / "services" / "billing" / "src" / "test" / "java" / "demo" / "PaymentServiceTest.java"
            generated_path = repository_path / "services" / "billing" / "target" / "generated-sources" / "demo" / "PaymentService.java"
            for path, source in {
                production_path: "package demo; public class PaymentService {}\n",
                test_path: "package demo; public class PaymentServiceTest {}\n",
                generated_path: "package demo; public class PaymentService {}\n",
            }.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(source, encoding="utf-8")

            with patch.object(repository_indexer, "enrich_java_semantic_edges") as enrich_java_semantic_edges:
                enrich_java_semantic_edges.side_effect = lambda graph, *, request: graph

                result = repository_indexer.index_repository(
                    repository_path,
                    embedding_client=_FakeEmbeddingClient(),
                    summary_client=_FakeSummaryClient(),
                )

        module_nodes = [node for node in result.graph.nodes if node.type == NodeType.MODULE]
        file_nodes = {node.properties["relative_path"]: node for node in result.graph.nodes if node.type == NodeType.FILE}
        entity_nodes = [node for node in result.graph.nodes if node.type == NodeType.ENTITY]

        self.assertEqual(result.indexed_file_count, 3)
        self.assertEqual(len(module_nodes), 1)
        self.assertEqual(module_nodes[0].properties["name"], "services:billing")
        self.assertEqual(module_nodes[0].properties["root_path"], "services/billing")
        self.assertEqual(module_nodes[0].properties["ecosystem"], "maven")
        self.assertEqual(module_nodes[0].properties["zone"], "prod")
        self.assertEqual(file_nodes["services/billing/src/main/java/demo/PaymentService.java"].properties["zone"], "prod")
        self.assertEqual(file_nodes["services/billing/src/test/java/demo/PaymentServiceTest.java"].properties["zone"], "test")
        self.assertEqual(
            file_nodes["services/billing/target/generated-sources/demo/PaymentService.java"].properties["zone"],
            GENERATED_ZONE,
        )
        self.assertTrue(
            any(
                node.properties["qualified_name"] == "demo.PaymentService"
                and node.properties["origin"] == "generated"
                for node in entity_nodes
            )
        )

    def test_incremental_index_builds_only_changed_file_summaries(self) -> None:
        with TemporaryDirectory() as workspace:
            repository_path = Path(workspace)
            stable_path = repository_path / "src" / "main" / "java" / "demo" / "Stable.java"
            changed_path = repository_path / "src" / "main" / "java" / "demo" / "Changed.java"
            added_path = repository_path / "src" / "main" / "java" / "demo" / "Added.java"
            stable_source = "package demo; public class Stable {}\n"
            changed_source = "package demo; public class Changed { public String value() { return \"new\"; } }\n"
            added_source = "package demo; public class Added {}\n"
            stable_path.parent.mkdir(parents=True)
            stable_path.write_text(stable_source, encoding="utf-8")
            changed_path.write_text(changed_source, encoding="utf-8")
            added_path.write_text(added_source, encoding="utf-8")
            previous_file_hashes = {
                "src/main/java/demo/Stable.java": _hash(stable_source),
                "src/main/java/demo/Changed.java": "sha256:old",
                "src/main/java/demo/Deleted.java": "sha256:deleted",
            }

            with patch.object(repository_indexer, "enrich_java_semantic_edges") as enrich_java_semantic_edges:
                enrich_java_semantic_edges.side_effect = lambda graph, **_kwargs: graph

                result = repository_indexer.index_repository_incremental(
                    repository_path,
                    previous_file_hashes=previous_file_hashes,
                    embedding_client=_FakeEmbeddingClient(),
                    summary_client=_FakeSummaryClient(),
                )

        described_target_ids = {
            edge.target_id
            for edge in result.graph.edges
            if edge.type == EdgeType.DESCRIBES
        }
        stable_file_ids = {
            node.id
            for node in result.graph.nodes
            if node.type == NodeType.FILE and node.properties["relative_path"] == "src/main/java/demo/Stable.java"
        }
        changed_file_ids = {
            node.id
            for node in result.graph.nodes
            if node.type == NodeType.FILE and node.properties["relative_path"] == "src/main/java/demo/Changed.java"
        }

        self.assertEqual(result.added_files, ["src/main/java/demo/Added.java"])
        self.assertEqual(result.modified_files, ["src/main/java/demo/Changed.java"])
        self.assertEqual(result.deleted_files, ["src/main/java/demo/Deleted.java"])
        self.assertEqual(result.skipped_files, ["src/main/java/demo/Stable.java"])
        self.assertFalse(stable_file_ids)
        self.assertTrue(changed_file_ids <= described_target_ids)
        self.assertEqual(enrich_java_semantic_edges.call_args.kwargs["source_file_paths"], {"src/main/java/demo/Added.java", "src/main/java/demo/Changed.java"})
        self.assertEqual(enrich_java_semantic_edges.call_args.kwargs["target_file_paths"], {"src/main/java/demo/Added.java", "src/main/java/demo/Changed.java"})

    def test_incremental_index_without_changes_skips_model_clients(self) -> None:
        with TemporaryDirectory() as workspace:
            repository_path = Path(workspace)
            source_path = repository_path / "src" / "main" / "java" / "demo" / "Stable.java"
            source = "package demo; public class Stable {}\n"
            source_path.parent.mkdir(parents=True)
            source_path.write_text(source, encoding="utf-8")

            result = repository_indexer.index_repository_incremental(
                repository_path,
                previous_file_hashes={"src/main/java/demo/Stable.java": _hash(source)},
            )

        self.assertEqual(result.changed_file_count, 0)
        self.assertEqual(result.indexed_file_count, 0)
        self.assertEqual(result.graph.nodes, [])
        self.assertEqual(result.graph.edges, [])

    def test_attach_retrieval_summaries_skips_empty_target_set(self) -> None:
        embedding_client = _FakeEmbeddingClient()
        summary_client = _FakeSummaryClient()
        graph = GraphIR()

        result = attach_retrieval_summaries(
            graph,
            embedding_client=embedding_client,
            summary_client=summary_client,
        )

        self.assertIs(result, graph)
        self.assertEqual(embedding_client.text_batches, [])
        self.assertEqual(summary_client.messages, [])

    def test_attach_retrieval_summaries_skips_existing_summary_nodes(self) -> None:
        embedding_client = _FakeEmbeddingClient()
        summary_client = _FakeSummaryClient()
        graph = GraphIR()
        module = _demo_module()
        summary = _retrieval_summary(module.module_id, text="已有摘要")
        graph.add_node(module)
        graph.add_node(summary)
        graph.add_edge(_describes_edge(summary.summary_id, module.module_id))

        result = attach_retrieval_summaries(
            graph,
            embedding_client=embedding_client,
            summary_client=summary_client,
        )

        self.assertIs(result, graph)
        self.assertEqual(embedding_client.text_batches, [])
        self.assertEqual(summary_client.messages, [])
        self.assertEqual([node.id for node in _summary_nodes(graph)], [summary.summary_id])

    def test_attach_retrieval_summaries_repairs_missing_existing_summary_edge(self) -> None:
        embedding_client = _FakeEmbeddingClient()
        summary_client = _FakeSummaryClient()
        graph = GraphIR()
        module = _demo_module()
        summary = _retrieval_summary(module.module_id, text="已有摘要")
        graph.add_node(module)
        graph.add_node(summary)

        attach_retrieval_summaries(
            graph,
            embedding_client=embedding_client,
            summary_client=summary_client,
        )

        self.assertEqual(embedding_client.text_batches, [])
        self.assertEqual(summary_client.messages, [])
        self.assertTrue(_has_describes_edge(graph, summary.summary_id, module.module_id))

    def test_attach_retrieval_summaries_regenerates_stale_source_summary(self) -> None:
        embedding_client = _FakeEmbeddingClient()
        summary_client = _FakeSummaryClient()
        graph = GraphIR()
        entity = _demo_entity(semantic_hash="sha256:new")
        stale_summary = _retrieval_summary(
            entity.entity_id,
            text="旧摘要",
            source_hash="sha256:old",
            embedding=[0.0, 1.0, 0.0],
        )
        graph.add_node(entity)
        graph.add_node(stale_summary)
        graph.add_edge(_describes_edge(stale_summary.summary_id, entity.entity_id))

        attach_retrieval_summaries(
            graph,
            embedding_client=embedding_client,
            summary_client=summary_client,
        )

        summary_nodes = _summary_nodes(graph)
        self.assertEqual(len(summary_nodes), 1)
        self.assertEqual(summary_nodes[0].properties["source_hash"], "sha256:new")
        self.assertNotEqual(summary_nodes[0].properties["text"], "旧摘要")
        self.assertEqual(len(embedding_client.text_batches), 1)
        self.assertEqual(len(summary_client.messages), 1)

    def test_attach_retrieval_summaries_regenerates_stale_model_summary(self) -> None:
        embedding_client = _FakeEmbeddingClient()
        summary_client = _FakeSummaryClient()
        graph = GraphIR()
        module = _demo_module()
        stale_summary = _retrieval_summary(
            module.module_id,
            text="旧模型摘要",
            summary_model="old-summary-model",
        )
        graph.add_node(module)
        graph.add_node(stale_summary)

        attach_retrieval_summaries(
            graph,
            embedding_client=embedding_client,
            summary_client=summary_client,
        )

        summary_nodes = _summary_nodes(graph)
        self.assertEqual(len(summary_nodes), 1)
        self.assertEqual(summary_nodes[0].properties["summary_model"], "summary-model")
        self.assertNotEqual(summary_nodes[0].properties["text"], "旧模型摘要")
        self.assertEqual(len(embedding_client.text_batches), 1)
        self.assertEqual(len(summary_client.messages), 1)

    def test_attach_retrieval_summaries_regenerates_stale_embedding_dimensions(self) -> None:
        embedding_client = _FakeEmbeddingClient(dimensions=4)
        summary_client = _FakeSummaryClient()
        graph = GraphIR()
        module = _demo_module()
        stale_summary = _retrieval_summary(
            module.module_id,
            text="旧维度摘要",
        )
        graph.add_node(module)
        graph.add_node(stale_summary)

        attach_retrieval_summaries(
            graph,
            embedding_client=embedding_client,
            summary_client=summary_client,
        )

        summary_nodes = _summary_nodes(graph)
        self.assertEqual(len(summary_nodes), 1)
        self.assertEqual(summary_nodes[0].properties["embedding_dimensions"], 4)
        self.assertNotEqual(summary_nodes[0].properties["text"], "旧维度摘要")
        self.assertEqual(len(embedding_client.text_batches), 1)
        self.assertEqual(len(summary_client.messages), 1)

    def test_attach_retrieval_summaries_generates_texts_concurrently(self) -> None:
        embedding_client = _FakeEmbeddingClient()
        summary_client = _ConcurrentSummaryClient(concurrent_requests=4)
        graph = GraphIR()
        modules = [
            Module(
                module_id=f"module:demo:module-{index}",
                name=f"module-{index}",
                root_path=f"module-{index}",
                ecosystem="maven",
                zone="prod",
            )
            for index in range(4)
        ]
        for module in modules:
            graph.add_node(module)

        attach_retrieval_summaries(
            graph,
            embedding_client=embedding_client,
            summary_client=summary_client,
        )

        self.assertGreater(summary_client.max_active_requests, 1)
        self.assertEqual(
            [node.properties["text"] for node in _summary_nodes(graph)],
            [f"摘要 {module.module_id}" for module in modules],
        )

    def test_attach_retrieval_summaries_ignores_batch_size_for_string_embedding_input(self) -> None:
        embedding_client = _FakeEmbeddingClient(input_mode=EmbeddingInputMode.STRING, batch_size=64)
        summary_client = _FakeSummaryClient()
        graph = GraphIR()
        first_module = Module(
            module_id="module:demo:first",
            name="first",
            root_path="first",
            ecosystem="maven",
            zone="prod",
        )
        second_module = Module(
            module_id="module:demo:second",
            name="second",
            root_path="second",
            ecosystem="maven",
            zone="prod",
        )
        graph.add_node(first_module)
        graph.add_node(second_module)

        attach_retrieval_summaries(
            graph,
            embedding_client=embedding_client,
            summary_client=summary_client,
        )

        self.assertEqual([len(batch) for batch in embedding_client.text_batches], [1, 1])


class _FakeEmbeddingClient:
    def __init__(
        self,
        *,
        dimensions: int = 3,
        input_mode: EmbeddingInputMode = EmbeddingInputMode.ARRAY,
        batch_size: int = 8,
    ) -> None:
        self.dimensions = dimensions
        self.config = EmbeddingConfig(
            provider="openai",
            format=EmbeddingFormat.OPENAI_EMBEDDINGS,
            model="text-embedding-3-small",
            api_key="fake-key",
            base_url=None,
            dimensions=dimensions,
            timeout_seconds=1,
            batch_size=batch_size,
            input_mode=input_mode,
        )
        self.text_batches: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.text_batches.append(texts)
        return [[1.0, *([0.0] * (self.dimensions - 1))] for _text in texts]


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
        self.messages: list[list[LLMMessage]] = []

    def generate_reply(self, messages: list[LLMMessage]) -> str:
        self.messages.append(messages)
        return f"模型摘要：{messages[-1].content[:20]}"


class _ConcurrentSummaryClient(_FakeSummaryClient):
    def __init__(self, *, concurrent_requests: int) -> None:
        super().__init__()
        self.config = LLMConfig(
            provider="openai",
            model="summary-model",
            api_key="fake-key",
            base_url=None,
            timeout_seconds=1,
            system_prompt="摘要提示",
            section=LLMConfigSection.SUMMARY,
            concurrent_requests=concurrent_requests,
        )
        self._lock = Lock()
        self._active_requests = 0
        self.max_active_requests = 0

    def generate_reply(self, messages: list[LLMMessage]) -> str:
        with self._lock:
            self._active_requests += 1
            self.max_active_requests = max(self.max_active_requests, self._active_requests)
        try:
            sleep(0.02)
            self.messages.append(messages)
            return f"摘要 {_prompt_node_id(messages[-1].content)}"
        finally:
            with self._lock:
                self._active_requests -= 1


def _demo_module() -> Module:
    return Module(
        module_id="module:demo:root",
        name="root",
        root_path=".",
        ecosystem="maven",
        zone="prod",
    )


def _demo_entity(*, semantic_hash: str) -> Entity:
    return Entity(
        entity_id="entity:demo:PaymentService",
        entity_key="java:demo.PaymentService",
        display_name="PaymentService",
        qualified_name="demo.PaymentService",
        kind_norm="class",
        kind_raw="class_declaration",
        origin="internal",
        semantic_hash=semantic_hash,
    )


def _retrieval_summary(
    target_node_id: str,
    *,
    text: str,
    source_hash: str = "sha256:existing",
    summary_model: str = "summary-model",
    embedding_model: str = "text-embedding-3-small",
    embedding: list[float] | None = None,
) -> Summary:
    resolved_embedding = embedding or [1.0, 0.0, 0.0]
    return Summary(
        summary_id=f"summary:{target_node_id}:retrieval",
        text=text,
        purpose="retrieval",
        source_hash=source_hash,
        summary_model=summary_model,
        embedding_model=embedding_model,
        embedding_dimensions=len(resolved_embedding),
        embedding=resolved_embedding,
    )


def _describes_edge(summary_id: str, target_id: str) -> GraphEdge:
    return GraphEdge.create(
        EdgeType.DESCRIBES,
        summary_id,
        target_id,
        kind="retrieval-summary",
        confidence=1.0,
    )


def _summary_nodes(graph: GraphIR) -> list[GraphNode]:
    return [node for node in graph.nodes if node.type == NodeType.SUMMARY]


def _has_describes_edge(graph: GraphIR, summary_id: str, target_id: str) -> bool:
    return any(
        edge.type == EdgeType.DESCRIBES
        and edge.source_id == summary_id
        and edge.target_id == target_id
        for edge in graph.edges
    )


def _prompt_node_id(prompt: str) -> str:
    for line in prompt.splitlines():
        if line.startswith("节点 ID："):
            return line.removeprefix("节点 ID：")
    raise AssertionError("摘要 Prompt 缺少节点 ID")


def _hash(source: str) -> str:
    return f"sha256:{hashlib.sha256(source.encode('utf-8')).hexdigest()}"
