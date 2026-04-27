from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from rigel_demo.indexing import repository_indexer
from rigel_demo.java.requests import DEFAULT_LSP_TIMEOUT_SECONDS


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

                result = repository_indexer.index_repository(repository_path)

        self.assertEqual(result.indexed_file_count, 1)
        self.assertEqual(enrich_java_semantic_edges.call_count, 1)
        self.assertNotIn("lsp_client", enrich_java_semantic_edges.call_args.kwargs)
        self.assertEqual(
            enrich_java_semantic_edges.call_args.kwargs["request"].lsp_timeout_seconds,
            DEFAULT_LSP_TIMEOUT_SECONDS,
        )
