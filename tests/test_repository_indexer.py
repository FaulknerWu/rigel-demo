import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from rigel_demo.graph import EdgeType, NodeType
from rigel_demo.embedding import EmbeddingConfig, EmbeddingFormat
from rigel_demo.project import repository_indexer
from rigel_demo.java.requests import DEFAULT_LSP_TIMEOUT_SECONDS, GENERATED_ZONE
from rigel_demo.llm import LLMConfig, LLMConfigSection, LLMMessage


class RepositoryIndexerTest(TestCase):
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
        return [[1.0, 0.0, 0.0] for _text in texts]


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
        return f"模型摘要：{messages[-1].content[:20]}"


def _hash(source: str) -> str:
    return f"sha256:{hashlib.sha256(source.encode('utf-8')).hexdigest()}"
