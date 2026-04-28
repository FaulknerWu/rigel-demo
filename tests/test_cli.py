from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from rigel_demo.cli import (
    DEFAULT_CONFIG_DOCUMENT,
    DEFAULT_GRAPH_NAME,
    database_artifact_exists,
    index_repository_workspace,
    init_repository,
)
from rigel_demo.core import EdgeType, GraphEdge, GraphIR, Module, Repository


class CliInitTest(TestCase):
    def test_init_repository_creates_default_config_without_database(self) -> None:
        with TemporaryDirectory() as workspace:
            repository_path = Path(workspace)

            result = init_repository(repository_path)

            config_path = repository_path / ".rigel" / "config.json"
            self.assertTrue(result.config_created)
            self.assertEqual(Path(result.config_path), config_path)
            self.assertEqual(json.loads(config_path.read_text(encoding="utf-8")), DEFAULT_CONFIG_DOCUMENT)
            self.assertFalse(database_artifact_exists(repository_path / ".rigel" / "falkordb.db"))

    def test_init_repository_keeps_existing_config(self) -> None:
        with TemporaryDirectory() as workspace:
            repository_path = Path(workspace)
            config_path = repository_path / ".rigel" / "config.json"
            config_path.parent.mkdir(parents=True)
            existing_config = {"llm": {"provider": "acme"}}
            config_path.write_text(json.dumps(existing_config, ensure_ascii=False) + "\n", encoding="utf-8")

            result = init_repository(repository_path)

            self.assertFalse(result.config_created)
            self.assertEqual(json.loads(config_path.read_text(encoding="utf-8")), existing_config)


class CliIndexTest(TestCase):
    def test_index_repository_workspace_requires_config(self) -> None:
        with TemporaryDirectory() as workspace:
            with self.assertRaisesRegex(FileNotFoundError, "rigel init"):
                index_repository_workspace(Path(workspace))

    def test_index_repository_workspace_writes_database_and_state(self) -> None:
        with TemporaryDirectory() as workspace:
            repository_path = Path(workspace)
            config_path = repository_path / ".rigel" / "config.json"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(json.dumps(DEFAULT_CONFIG_DOCUMENT, ensure_ascii=False) + "\n", encoding="utf-8")
            graph = _demo_graph()

            with patch("rigel_demo.indexing.repository_indexer.index_repository") as index_repository:
                index_repository.return_value = SimpleNamespace(graph=graph, indexed_file_count=1)

                result = index_repository_workspace(repository_path)

            database_path = repository_path / ".rigel" / "falkordb.db"
            state_path = repository_path / ".rigel" / "rigel.json"
            self.assertEqual(Path(result.database_path), database_path)
            self.assertTrue(database_artifact_exists(database_path))
            self.assertEqual(result.indexed_file_count, 1)
            self.assertEqual(result.graph_node_count, 2)
            self.assertEqual(result.graph_edge_count, 1)
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["graph_name"], DEFAULT_GRAPH_NAME)
            self.assertEqual(state["database_path"], str(database_path))
            self.assertIn("indexed_at", state)


def _demo_graph() -> GraphIR:
    graph = GraphIR()
    repository = Repository(repo_id="repo:demo", name="demo")
    module = Module(
        module_id="module:demo:root",
        name="root",
        root_path=".",
        ecosystem="maven",
        zone="prod",
    )
    graph.add_node(repository)
    graph.add_node(module)
    graph.add_edge(
        GraphEdge.create(EdgeType.CONTAINS, repository.repo_id, module.module_id, kind="physical-membership")
    )
    return graph
