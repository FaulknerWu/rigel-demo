from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from rigel_demo.core import EdgeType, NodeType
from rigel_demo.embedding import EmbeddingConfig, EmbeddingFormat
from rigel_demo.indexing import repository_indexer
from rigel_demo.java.requests import DEFAULT_LSP_TIMEOUT_SECONDS
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
