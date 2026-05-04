from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from rigel_demo.embedding import (
    EmbeddingConfig,
    EmbeddingConfigurationError,
    EmbeddingFormat,
    EmbeddingInputMode,
    EmbeddingResponseError,
    RigelEmbedding,
    build_rigel_embedding,
)


class EmbeddingConfigTest(TestCase):
    def test_embedding_config_reads_openai_model_and_dimensions(self) -> None:
        with TemporaryDirectory() as workspace:
            repository_path = _write_embedding_config(
                Path(workspace),
                {
                    "provider": "openai",
                    "format": "openai_embeddings",
                    "model": "text-embedding-3-small",
                    "api_key": "embedding-key",
                    "base_url": None,
                    "dimensions": 512,
                    "timeout_seconds": 60,
                    "batch_size": 16,
                    "input_mode": "array",
                },
            )

            config = EmbeddingConfig.from_repository(repository_path)

        self.assertEqual(config.provider, "openai")
        self.assertEqual(config.format, EmbeddingFormat.OPENAI_EMBEDDINGS)
        self.assertEqual(config.model, "text-embedding-3-small")
        self.assertEqual(config.api_key, "embedding-key")
        self.assertIsNone(config.base_url)
        self.assertEqual(config.dimensions, 512)
        self.assertEqual(config.timeout_seconds, 60)
        self.assertEqual(config.batch_size, 16)
        self.assertEqual(config.input_mode, EmbeddingInputMode.ARRAY)

    def test_embedding_config_requires_template_fields(self) -> None:
        with TemporaryDirectory() as workspace:
            repository_path = _write_embedding_config(
                Path(workspace),
                {
                    "provider": "openai",
                    "format": "openai_embeddings",
                    "model": "text-embedding-3-small",
                    "api_key": "embedding-key",
                    "base_url": None,
                    "dimensions": 512,
                    "batch_size": 16,
                    "input_mode": "array",
                },
            )

            with self.assertRaisesRegex(EmbeddingConfigurationError, "embedding.timeout_seconds"):
                EmbeddingConfig.from_repository(repository_path)

    def test_embedding_config_defaults_input_mode_to_array(self) -> None:
        with TemporaryDirectory() as workspace:
            repository_path = _write_embedding_config(
                Path(workspace),
                {
                    "provider": "openai",
                    "format": "openai_embeddings",
                    "model": "Qwen3-Embedding-8B",
                    "api_key": "embedding-key",
                    "base_url": "https://ai.gitee.com/v1",
                    "dimensions": 2000,
                    "timeout_seconds": 60,
                    "batch_size": 25,
                },
            )

            config = EmbeddingConfig.from_repository(repository_path)

        self.assertEqual(config.input_mode, EmbeddingInputMode.ARRAY)


class RigelEmbeddingTest(TestCase):
    def test_requires_explicit_openai_client(self) -> None:
        with self.assertRaisesRegex(EmbeddingConfigurationError, "openai_client"):
            RigelEmbedding(_embedding_config())

    def test_build_rigel_embedding_creates_configured_openai_client(self) -> None:
        with patch("rigel_demo.embedding.client.OpenAI", _FakeOpenAIClient), patch(
            "rigel_demo.embedding.client.DefaultHttpxClient",
            _FakeHttpxClient,
        ):
            embedding = build_rigel_embedding(_embedding_config())

        self.assertIsInstance(embedding, RigelEmbedding)
        self.assertEqual(
            embedding._client.kwargs,
            {
                "api_key": "embedding-key",
                "base_url": None,
                "timeout": 30,
                "http_client": _FakeHttpxClient(trust_env=False),
            },
        )

    def test_embed_texts_uses_openai_embeddings_request_shape(self) -> None:
        fake_client = _FakeOpenAIEmbeddingClient()
        embedding = RigelEmbedding(_embedding_config(), openai_client=fake_client)

        vectors = embedding.embed_texts(["PaymentService", "OrderRepository"])

        self.assertEqual(vectors, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        self.assertEqual(
            fake_client.embeddings.request_body,
            {
                "model": "text-embedding-3-small",
                "input": ["PaymentService", "OrderRepository"],
                "encoding_format": "float",
                "dimensions": 512,
            },
        )

    def test_embed_texts_uses_response_data_order_without_index_validation(self) -> None:
        fake_client = _FakeOpenAIEmbeddingClient(
            response_data=[
                SimpleNamespace(index=0, embedding=[1.0, 0.0, 0.0]),
                SimpleNamespace(index=0, embedding=[0.0, 1.0, 0.0]),
            ]
        )
        embedding = RigelEmbedding(_embedding_config(), openai_client=fake_client)

        vectors = embedding.embed_texts(["PaymentService", "OrderRepository"])

        self.assertEqual(vectors, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    def test_embed_texts_rejects_blank_input_text(self) -> None:
        fake_client = _FakeOpenAIEmbeddingClient()
        embedding = RigelEmbedding(_embedding_config(), openai_client=fake_client)

        with self.assertRaisesRegex(EmbeddingResponseError, "输入文本不能为空"):
            embedding.embed_texts(["PaymentService", "   "])

        self.assertEqual(fake_client.embeddings.request_body, {})

    def test_embed_texts_strips_input_text_before_request(self) -> None:
        fake_client = _FakeOpenAIEmbeddingClient()
        embedding = RigelEmbedding(_embedding_config(), openai_client=fake_client)

        vectors = embedding.embed_texts(["  PaymentService  ", "  OrderRepository  "])

        self.assertEqual(vectors, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        self.assertEqual(fake_client.embeddings.request_body["input"], ["PaymentService", "OrderRepository"])

    def test_embed_texts_supports_string_input_mode(self) -> None:
        fake_client = _FakeOpenAIEmbeddingClient()
        embedding = RigelEmbedding(
            _embedding_config(input_mode=EmbeddingInputMode.STRING, batch_size=64),
            openai_client=fake_client,
        )

        vectors = embedding.embed_texts(["PaymentService", "OrderRepository"])

        self.assertEqual(vectors, [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        self.assertEqual(
            fake_client.embeddings.request_bodies,
            [
                {
                    "model": "text-embedding-3-small",
                    "input": "PaymentService",
                    "encoding_format": "float",
                    "dimensions": 512,
                },
                {
                    "model": "text-embedding-3-small",
                    "input": "OrderRepository",
                    "encoding_format": "float",
                    "dimensions": 512,
                },
            ],
        )


def _write_embedding_config(repository_path: Path, config: dict[str, object]) -> Path:
    config_path = repository_path / ".rigel" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps({"embedding": config}, ensure_ascii=False) + "\n", encoding="utf-8")
    return repository_path


def _embedding_config(
    *,
    input_mode: EmbeddingInputMode = EmbeddingInputMode.ARRAY,
    batch_size: int = 8,
) -> EmbeddingConfig:
    return EmbeddingConfig(
        provider="openai",
        format=EmbeddingFormat.OPENAI_EMBEDDINGS,
        model="text-embedding-3-small",
        api_key="embedding-key",
        base_url=None,
        dimensions=512,
        timeout_seconds=30,
        batch_size=batch_size,
        input_mode=input_mode,
    )


class _FakeOpenAIEmbeddingClient:
    def __init__(self, response_data: list[SimpleNamespace] | None = None) -> None:
        self.embeddings = _FakeEmbeddingsResource(response_data)


class _FakeOpenAIClient:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.embeddings = _FakeEmbeddingsResource()


class _FakeHttpxClient:
    def __init__(self, *, trust_env: bool) -> None:
        self.trust_env = trust_env

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _FakeHttpxClient) and self.trust_env == other.trust_env


class _FakeEmbeddingsResource:
    def __init__(self, response_data: list[SimpleNamespace] | None = None) -> None:
        self.request_body: dict[str, object] = {}
        self.request_bodies: list[dict[str, object]] = []
        self.response_data = response_data

    def create(self, **request_body: object) -> SimpleNamespace:
        self.request_body = request_body
        self.request_bodies.append(request_body)
        if self.response_data is not None:
            return SimpleNamespace(data=self.response_data)
        if isinstance(request_body["input"], str):
            return SimpleNamespace(
                data=[
                    SimpleNamespace(index=0, embedding=[1.0, 0.0, 0.0]),
                ]
            )
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=0, embedding=[1.0, 0.0, 0.0]),
                SimpleNamespace(index=1, embedding=[0.0, 1.0, 0.0]),
            ]
        )
