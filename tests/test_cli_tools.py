from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rigel_demo import cli


def test_vector_search_seeds_cli_outputs_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.chdir(tmp_path)
    write_workspace_files(tmp_path)
    fake_repository_executor = FakeRepositoryExecutor(
        FakeExecutor(
            {
                "tool": "vector_search_seeds",
                "items": [{"node": {"id": "node-a"}}],
                "warnings": [],
            }
        )
    )
    monkeypatch.setattr(cli, "build_repository_tool_executor", fake_builder(fake_repository_executor))

    exit_code = cli.main(["tools", "vector-search-seeds", "--query", "  付款服务  ", "--query", "订单"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert fake_repository_executor.executor.calls == [
        ("vector_search_seeds", {"query_texts": ["付款服务", "订单"]})
    ]
    assert json.loads(captured.out) == {
        "status": "success",
        "tool": "vector_search_seeds",
        "items": [{"node": {"id": "node-a"}}],
        "warnings": [],
    }


def test_expand_neighbors_cli_uses_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.chdir(tmp_path)
    write_workspace_files(tmp_path)
    fake_repository_executor = FakeRepositoryExecutor(
        FakeExecutor(
            {
                "tool": "expand_neighbors",
                "items": [{"edge": {"id": "edge-a"}}],
                "warnings": [],
            }
        )
    )
    monkeypatch.setattr(cli, "build_repository_graph_executor", fake_builder(fake_repository_executor))

    exit_code = cli.main(["tools", "expand-neighbors", "--node-id", "node-a", "--node-id", "node-b"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert fake_repository_executor.executor.calls == [
        ("expand_neighbors", {"node_ids": ["node-a", "node-b"], "direction": "both"})
    ]
    assert json.loads(captured.out)["tool"] == "expand_neighbors"


def test_expand_neighbors_cli_accepts_repeated_edge_types(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    write_workspace_files(tmp_path)
    fake_repository_executor = FakeRepositoryExecutor(FakeExecutor({"tool": "expand_neighbors", "items": [], "warnings": []}))
    monkeypatch.setattr(cli, "build_repository_graph_executor", fake_builder(fake_repository_executor))

    exit_code = cli.main(
        [
            "tools",
            "expand-neighbors",
            "--node-id",
            "node-a",
            "--direction",
            "incoming",
            "--edge-type",
            "CONTAINS",
            "--edge-type",
            "DEPENDS_ON",
        ]
    )

    capsys.readouterr()
    assert exit_code == 0
    assert fake_repository_executor.executor.calls == [
        (
            "expand_neighbors",
            {
                "node_ids": ["node-a"],
                "direction": "incoming",
                "edge_types": ["CONTAINS", "DEPENDS_ON"],
            },
        )
    ]


def test_query_relation_cli_requires_relation_type(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    write_workspace_files(tmp_path)
    fake_repository_executor = FakeRepositoryExecutor(FakeExecutor({"tool": "query_relation", "items": [], "warnings": []}))
    monkeypatch.setattr(cli, "build_repository_graph_executor", fake_builder(fake_repository_executor))

    exit_code = cli.main(
        [
            "tools",
            "query-relation",
            "--node-id",
            "node-a",
            "--relation-type",
            "SPECIALIZES",
            "--direction",
            "outgoing",
        ]
    )

    capsys.readouterr()
    assert exit_code == 0
    assert fake_repository_executor.executor.calls == [
        (
            "query_relation",
            {
                "node_ids": ["node-a"],
                "relation_type": "SPECIALIZES",
                "direction": "outgoing",
            },
        )
    ]


@pytest.mark.parametrize(
    "argv",
    [
        ["tools", "vector-search-seeds", "--query", ""],
        ["tools", "expand-neighbors", "--node-id", ""],
    ],
)
def test_cli_rejects_empty_repeated_arguments(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(argv)

    captured = capsys.readouterr()
    assert error.value.code == 2
    assert captured.out == ""
    assert "参数错误" in captured.err


@pytest.mark.parametrize(
    "argv",
    [
        ["tools", "expand-neighbors", "--node-id", "node-a", "--direction", "sideways"],
        ["tools", "query-relation", "--node-id", "node-a", "--relation-type", "HAS_ANCHOR"],
    ],
)
def test_cli_rejects_invalid_tool_options(
    argv: list[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    write_workspace_files(tmp_path)

    exit_code = cli.main(argv)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err.strip()


def write_workspace_files(repository_path: Path) -> None:
    workspace_path = repository_path / ".rigel"
    workspace_path.mkdir()
    (workspace_path / "config.json").write_text("{}", encoding="utf-8")
    (workspace_path / "falkordb.db").write_text("", encoding="utf-8")


def fake_builder(repository_executor: "FakeRepositoryExecutor") -> Any:
    def build(*args: object, **kwargs: object) -> FakeRepositoryExecutor:
        return repository_executor

    return build


class FakeRepositoryExecutor:
    def __init__(self, executor: "FakeExecutor") -> None:
        self.executor = executor

    def __enter__(self) -> "FakeRepositoryExecutor":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        return None


class FakeExecutor:
    def __init__(self, result: dict[str, object]) -> None:
        self._result = result
        self.calls: list[tuple[str, dict[str, object]]] = []

    def vector_search_seeds(self, args: dict[str, object]) -> dict[str, object]:
        self.calls.append(("vector_search_seeds", args))
        return self._result

    def expand_neighbors(self, args: dict[str, object]) -> dict[str, object]:
        self.calls.append(("expand_neighbors", args))
        return self._result

    def query_relation(self, args: dict[str, object]) -> dict[str, object]:
        self.calls.append(("query_relation", args))
        return self._result
