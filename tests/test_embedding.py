from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase

from rigel_demo.embedding import EmbeddingConfig, EmbeddingConfigurationError, EmbeddingFormat, RigelEmbedding


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
                },
            )

            with self.assertRaisesRegex(EmbeddingConfigurationError, "embedding.timeout_seconds"):
                EmbeddingConfig.from_repository(repository_path)


class RigelEmbeddingTest(TestCase):
    def test_embed_texts_uses_openai_embeddings_request_shape(self) -> None:
        fake_client = _FakeOpenAIEmbeddingClient()
        config = EmbeddingConfig(
            provider="openai",
            format=EmbeddingFormat.OPENAI_EMBEDDINGS,
            model="text-embedding-3-small",
            api_key="embedding-key",
            base_url=None,
            dimensions=512,
            timeout_seconds=30,
            batch_size=8,
        )
        embedding = RigelEmbedding(config, openai_client=fake_client)

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


def _write_embedding_config(repository_path: Path, config: dict[str, object]) -> Path:
    config_path = repository_path / ".rigel" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps({"embedding": config}, ensure_ascii=False) + "\n", encoding="utf-8")
    return repository_path


class _FakeOpenAIEmbeddingClient:
    def __init__(self) -> None:
        self.embeddings = _FakeEmbeddingsResource()


class _FakeEmbeddingsResource:
    def __init__(self) -> None:
        self.request_body: dict[str, object] = {}

    def create(self, **request_body: object) -> SimpleNamespace:
        self.request_body = request_body
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=0, embedding=[1.0, 0.0, 0.0]),
                SimpleNamespace(index=1, embedding=[0.0, 1.0, 0.0]),
            ]
        )
