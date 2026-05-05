from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rigel_demo.config import RerankConfig, RerankConfigurationError
from rigel_demo.config.document import RIGEL_CONFIG_FILE_NAME, RIGEL_WORKSPACE_DIRECTORY_NAME
from rigel_demo.rerank import RigelReranker


def test_rerank_config_requires_section(tmp_path: Path) -> None:
    write_config(tmp_path, {})

    with pytest.raises(RerankConfigurationError, match="rerank"):
        RerankConfig.from_repository(tmp_path)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("base_url", "rerank.base_url"),
        ("model", "rerank.model"),
    ],
)
def test_rerank_config_requires_strings(tmp_path: Path, field: str, message: str) -> None:
    config = valid_rerank_config()
    config[field] = ""
    write_config(tmp_path, {"rerank": config})

    with pytest.raises(RerankConfigurationError, match=message):
        RerankConfig.from_repository(tmp_path)


def test_rerank_config_rejects_invalid_top_n(tmp_path: Path) -> None:
    config = valid_rerank_config()
    config["top_n"] = 0
    write_config(tmp_path, {"rerank": config})

    with pytest.raises(RerankConfigurationError, match="rerank.top_n"):
        RerankConfig.from_repository(tmp_path)


def test_reranker_posts_expected_request_and_parses_results() -> None:
    session = FakeSession(
        {
            "results": [
                {"index": 1, "relevance_score": 0.94},
                {"index": 0, "relevance_score": 0.41},
            ]
        }
    )
    reranker = RigelReranker(
        RerankConfig(
            provider="gitee_ai",
            base_url="https://ai.gitee.com/v1",
            model="Qwen3-Reranker-4B",
            api_key="token",
            timeout_seconds=12,
            top_n=5,
            candidate_limit_per_query=8,
            failover_enabled=True,
        ),
        http_session=session,
    )

    results = reranker.rerank(query="读 CSV", documents=["pandas read_csv", "json dump"], top_n=2)

    assert session.request["url"] == "https://ai.gitee.com/v1/rerank"
    assert session.request["headers"]["Authorization"] == "Bearer token"
    assert session.request["headers"]["Content-Type"] == "application/json"
    assert session.request["headers"]["X-Failover-Enabled"] == "true"
    assert session.request["json"] == {
        "model": "Qwen3-Reranker-4B",
        "query": "读 CSV",
        "documents": ["pandas read_csv", "json dump"],
        "top_n": 2,
    }
    assert [(result.index, result.relevance_score) for result in results] == [(1, 0.94), (0, 0.41)]


def valid_rerank_config() -> dict[str, object]:
    return {
        "provider": "gitee_ai",
        "base_url": "https://ai.gitee.com/v1",
        "model": "Qwen3-Reranker-4B",
        "api_key": "token",
        "timeout_seconds": 60,
        "top_n": 5,
        "candidate_limit_per_query": 8,
        "failover_enabled": False,
    }


def write_config(repository_path: Path, config: dict[str, object]) -> None:
    config_path = repository_path / RIGEL_WORKSPACE_DIRECTORY_NAME / RIGEL_CONFIG_FILE_NAME
    config_path.parent.mkdir()
    config_path.write_text(json.dumps(config), encoding="utf-8")


class FakeSession:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload
        self.request: dict[str, Any] = {}

    def post(self, url: str, *, headers: dict[str, str], json: dict[str, object], timeout: float) -> "FakeResponse":
        self.request = {
            "url": url,
            "headers": headers,
            "json": json,
            "timeout": timeout,
        }
        return FakeResponse(self._payload)


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._payload
