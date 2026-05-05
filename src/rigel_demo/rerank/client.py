"""Gitee AI 兼容 Rerank 客户端。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import requests

from rigel_demo.config import RerankConfig, RerankConfigurationError


class RerankRequestError(RuntimeError):
    """Rerank 远程请求失败。"""


class RerankResponseError(RuntimeError):
    """Rerank 返回内容无法解析。"""


@dataclass(frozen=True, slots=True)
class RerankResult:
    """单条 Rerank 结果。"""

    index: int
    relevance_score: float


class RigelReranker:
    """统一封装 Rerank 模型调用。"""

    def __init__(self, config: RerankConfig, *, http_session: Any | None = None) -> None:
        if config.provider != "gitee_ai":
            raise RerankConfigurationError(f"不支持的 Rerank 提供商：{config.provider}")
        self._config = config
        self._http_session = http_session or requests.Session()

    @property
    def config(self) -> RerankConfig:
        """返回当前 Rerank 配置。"""

        return self._config

    def rerank(self, *, query: str, documents: Sequence[str], top_n: int) -> list[RerankResult]:
        """按相关性重排候选文档。"""

        normalized_query = query.strip()
        normalized_documents = _normalize_documents(documents)
        if not normalized_query:
            raise RerankResponseError("Rerank 查询文本不能为空")
        if top_n <= 0:
            raise RerankResponseError("Rerank top_n 必须大于 0")

        payload = {
            "model": self._config.model,
            "query": normalized_query,
            "documents": normalized_documents,
            "top_n": min(top_n, len(normalized_documents)),
        }
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }
        if self._config.failover_enabled:
            headers["X-Failover-Enabled"] = "true"

        try:
            response = self._http_session.post(
                f"{self._config.base_url}/rerank",
                headers=headers,
                json=payload,
                timeout=self._config.timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as error:
            raise RerankRequestError(f"Rerank 调用失败：{error}") from error

        try:
            response_payload = response.json()
        except ValueError as error:
            raise RerankResponseError("Rerank 返回内容不是合法 JSON") from error

        return _parse_rerank_results(response_payload, document_count=len(normalized_documents))


def build_rigel_reranker(config: RerankConfig) -> RigelReranker:
    """构造生产路径使用的 Rerank 客户端。"""

    return RigelReranker(config)


def _normalize_documents(documents: Sequence[str]) -> list[str]:
    normalized_documents: list[str] = []
    for document in documents:
        normalized_document = document.strip()
        if not normalized_document:
            raise RerankResponseError("Rerank 候选文档不能为空")
        normalized_documents.append(normalized_document)
    if not normalized_documents:
        raise RerankResponseError("Rerank 候选文档列表不能为空")
    return normalized_documents


def _parse_rerank_results(payload: object, *, document_count: int) -> list[RerankResult]:
    if not isinstance(payload, Mapping):
        raise RerankResponseError("Rerank 返回根节点必须是对象")
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raise RerankResponseError("Rerank 返回必须包含 results 数组")

    results: list[RerankResult] = []
    for raw_result in raw_results:
        if not isinstance(raw_result, Mapping):
            raise RerankResponseError("Rerank results 条目必须是对象")
        index = _read_result_index(raw_result, document_count=document_count)
        relevance_score = _read_relevance_score(raw_result)
        results.append(RerankResult(index=index, relevance_score=relevance_score))
    return results


def _read_result_index(result: Mapping[str, object], *, document_count: int) -> int:
    value = result.get("index")
    if isinstance(value, bool) or not isinstance(value, int):
        raise RerankResponseError("Rerank results.index 必须是整数")
    if value < 0 or value >= document_count:
        raise RerankResponseError("Rerank results.index 超出 documents 范围")
    return value


def _read_relevance_score(result: Mapping[str, object]) -> float:
    value = result.get("relevance_score")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RerankResponseError("Rerank results.relevance_score 必须是数字")
    return float(value)
