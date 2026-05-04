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
    FrontendBuildResult,
    WebConfig,
    WebConfigurationError,
    build_frontend,
    database_artifact_exists,
    index_repository_workspace,
    init_repository,
    main,
)
from rigel_demo.core import EdgeType, GraphEdge, GraphIR, Module, Repository
from rigel_demo.storage import FalkorDBConfig, FalkorDBStore


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
            existing_config = {"chat": {"provider": "acme"}}
            config_path.write_text(json.dumps(existing_config, ensure_ascii=False) + "\n", encoding="utf-8")

            result = init_repository(repository_path)

            self.assertFalse(result.config_created)
            self.assertEqual(json.loads(config_path.read_text(encoding="utf-8")), existing_config)


class CliIndexTest(TestCase):
    def test_index_repository_workspace_requires_config(self) -> None:
        with TemporaryDirectory() as workspace:
            with self.assertRaisesRegex(FileNotFoundError, "rigel init"):
                index_repository_workspace(Path(workspace))

    def test_incremental_index_requires_existing_database(self) -> None:
        with TemporaryDirectory() as workspace:
            repository_path = Path(workspace)
            config_path = repository_path / ".rigel" / "config.json"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(json.dumps(DEFAULT_CONFIG_DOCUMENT, ensure_ascii=False) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(FileNotFoundError, "可增量索引"):
                index_repository_workspace(repository_path, incremental=True)

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
            self.assertEqual(state["last_index_mode"], "full")

    def test_incremental_index_workspace_updates_database_and_state(self) -> None:
        with TemporaryDirectory() as workspace:
            repository_path = Path(workspace)
            config_path = repository_path / ".rigel" / "config.json"
            database_path = repository_path / ".rigel" / "falkordb.db"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(json.dumps(DEFAULT_CONFIG_DOCUMENT, ensure_ascii=False) + "\n", encoding="utf-8")
            FalkorDBStore.connect(
                FalkorDBConfig(graph_name=DEFAULT_GRAPH_NAME, database_path=str(database_path))
            ).upsert_graph(_demo_graph())

            with patch("rigel_demo.indexing.repository_indexer.index_repository_incremental") as index_repository_incremental:
                index_repository_incremental.return_value = SimpleNamespace(
                    graph=GraphIR(),
                    added_files=["src/main/java/demo/Added.java"],
                    modified_files=[],
                    deleted_files=[],
                    skipped_files=["src/main/java/demo/Stable.java"],
                    indexed_file_count=1,
                )

                result = index_repository_workspace(repository_path, incremental=True)

            state = json.loads((repository_path / ".rigel" / "rigel.json").read_text(encoding="utf-8"))

        self.assertEqual(result.index_mode, "incremental")
        self.assertEqual(result.added_files, ("src/main/java/demo/Added.java",))
        self.assertEqual(result.skipped_file_count, 1)
        self.assertEqual(state["last_index_mode"], "incremental")
        self.assertEqual(state["last_incremental_result"]["added_files"], ["src/main/java/demo/Added.java"])
        self.assertEqual(state["last_incremental_result"]["skipped_file_count"], 1)


class CliWebConfigTest(TestCase):
    def test_web_config_reads_repository_config(self) -> None:
        with TemporaryDirectory() as workspace:
            repository_path = Path(workspace)
            _write_config(
                repository_path,
                {
                    **DEFAULT_CONFIG_DOCUMENT,
                    "web": {
                        "host": "0.0.0.0",
                        "port": 8080,
                        "open_browser": False,
                    },
                },
            )

            config = WebConfig.from_repository(repository_path)

        self.assertEqual(config.host, "0.0.0.0")
        self.assertEqual(config.port, 8080)
        self.assertFalse(config.open_browser)

    def test_web_config_rejects_invalid_port(self) -> None:
        with TemporaryDirectory() as workspace:
            repository_path = Path(workspace)
            _write_config(
                repository_path,
                {
                    **DEFAULT_CONFIG_DOCUMENT,
                    "web": {
                        "host": "127.0.0.1",
                        "port": 0,
                        "open_browser": True,
                    },
                },
            )

            with self.assertRaisesRegex(WebConfigurationError, "web.port"):
                WebConfig.from_repository(repository_path)

    def test_web_command_uses_configured_server_options(self) -> None:
        with TemporaryDirectory() as workspace:
            repository_path = Path(workspace)
            static_path = repository_path / ".rigel" / "web" / "static"
            static_path.joinpath("assets").mkdir(parents=True, exist_ok=True)
            static_path.joinpath("index.html").write_text("<!doctype html><div id=\"root\"></div>\n", encoding="utf-8")
            repository_path.joinpath(".rigel", "rigel.json").write_text('{"graph_name": "rigel"}\n', encoding="utf-8")
            _write_config(
                repository_path,
                {
                    **DEFAULT_CONFIG_DOCUMENT,
                    "web": {
                        "host": "0.0.0.0",
                        "port": 8080,
                        "open_browser": False,
                    },
                },
            )

            with (
                patch("rigel_demo.cli.Path.cwd", return_value=repository_path),
                patch("rigel_demo.cli.database_artifact_exists", return_value=True),
                patch(
                    "rigel_demo.cli.build_frontend",
                    return_value=FrontendBuildResult(success=True, static_path=static_path),
                ),
                patch("rigel_demo.cli.webbrowser.open") as open_browser,
                patch("uvicorn.run") as uvicorn_run,
            ):
                exit_code = main(["web"])

        self.assertEqual(exit_code, 0)
        open_browser.assert_not_called()
        self.assertEqual(uvicorn_run.call_args.kwargs["host"], "0.0.0.0")
        self.assertEqual(uvicorn_run.call_args.kwargs["port"], 8080)


class CliFrontendBuildTest(TestCase):
    def test_build_frontend_explains_npm_install_requirement(self) -> None:
        with TemporaryDirectory() as workspace:
            static_path = Path(workspace) / ".rigel" / "web" / "static"

            with patch("rigel_demo.cli.shutil.which", return_value=None):
                result = build_frontend(static_path)

        self.assertFalse(result.success)
        self.assertIn("请先安装 Node.js/npm", result.message)
        self.assertIn("npm --version", result.message)


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


def _write_config(repository_path: Path, config: dict[str, object]) -> None:
    config_path = repository_path / ".rigel" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, ensure_ascii=False) + "\n", encoding="utf-8")
