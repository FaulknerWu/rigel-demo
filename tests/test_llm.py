from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase

from rigel_demo.llm import (
    DEFAULT_GOOGLE_BASE_URL,
    LLMConfig,
    LLMConfigurationError,
    LLMFormat,
    LLMMessage,
    RigelLLM,
)


class LLMConfigTest(TestCase):
    def test_google_provider_uses_native_generate_content_defaults(self) -> None:
        with TemporaryDirectory() as workspace:
            repository_path = _write_llm_config(
                Path(workspace),
                {
                    "provider": "google",
                    "model": "gemini-3-flash-preview",
                    "api_key": "gemini-key",
                },
            )

            config = LLMConfig.from_repository(repository_path)

        self.assertEqual(config.provider, "google")
        self.assertEqual(config.format, LLMFormat.GOOGLE_GENERATE_CONTENT)
        self.assertEqual(config.model, "gemini-3-flash-preview")
        self.assertEqual(config.api_key, "gemini-key")
        self.assertEqual(config.base_url, DEFAULT_GOOGLE_BASE_URL)

    def test_openai_responses_config_supports_custom_base_url(self) -> None:
        with TemporaryDirectory() as workspace:
            repository_path = _write_llm_config(
                Path(workspace),
                {
                    "provider": "openai",
                    "format": "openai_responses",
                    "model": "gpt-custom",
                    "api_key": "openai-key",
                    "base_url": "https://example.test/v1",
                    "max_output_tokens": 1024,
                },
            )

            config = LLMConfig.from_repository(repository_path)

        self.assertEqual(config.provider, "openai")
        self.assertEqual(config.format, LLMFormat.OPENAI_RESPONSES)
        self.assertEqual(config.base_url, "https://example.test/v1")
        self.assertEqual(config.max_output_tokens, 1024)

    def test_custom_provider_requires_base_url(self) -> None:
        with TemporaryDirectory() as workspace:
            repository_path = _write_llm_config(
                Path(workspace),
                {
                    "provider": "acme",
                    "format": "openai_chat",
                    "model": "acme-chat",
                    "api_key": "acme-key",
                },
            )

            with self.assertRaisesRegex(LLMConfigurationError, "llm.base_url"):
                LLMConfig.from_repository(repository_path)

    def test_custom_provider_accepts_configured_format_key_and_base_url(self) -> None:
        with TemporaryDirectory() as workspace:
            repository_path = _write_llm_config(
                Path(workspace),
                {
                    "provider": "acme",
                    "format": "openai_chat",
                    "model": "acme-chat",
                    "api_key": "acme-key",
                    "base_url": "https://acme.example/v1",
                },
            )

            config = LLMConfig.from_repository(repository_path)

        self.assertEqual(config.provider, "acme")
        self.assertEqual(config.format, LLMFormat.OPENAI_CHAT)
        self.assertEqual(config.api_key, "acme-key")
        self.assertEqual(config.base_url, "https://acme.example/v1")

    def test_missing_model_fails_fast(self) -> None:
        with TemporaryDirectory() as workspace:
            repository_path = _write_llm_config(Path(workspace), {"api_key": "test-key"})

            with self.assertRaisesRegex(LLMConfigurationError, "llm.model"):
                LLMConfig.from_repository(repository_path)

    def test_missing_config_file_fails_fast(self) -> None:
        with TemporaryDirectory() as workspace:
            with self.assertRaisesRegex(LLMConfigurationError, ".rigel/config.json"):
                LLMConfig.from_repository(Path(workspace))


class RigelLLMTest(TestCase):
    def test_generate_reply_uses_openai_responses_format(self) -> None:
        fake_client = _FakeOpenAIClient()
        config = _config(format=LLMFormat.OPENAI_RESPONSES)
        llm = RigelLLM(config, openai_client=fake_client)

        reply = llm.generate_reply([LLMMessage(role="user", content="分析 Service")])

        self.assertEqual(reply, "Responses 回复")
        self.assertEqual(fake_client.responses.request_body["model"], "test-model")
        self.assertEqual(fake_client.responses.request_body["instructions"], "系统提示")
        self.assertEqual(fake_client.responses.request_body["input"], [{"role": "user", "content": "分析 Service"}])

    def test_generate_reply_uses_openai_chat_format(self) -> None:
        fake_client = _FakeOpenAIClient()
        config = _config(format=LLMFormat.OPENAI_CHAT)
        llm = RigelLLM(config, openai_client=fake_client)

        reply = llm.generate_reply([LLMMessage(role="user", content="分析 Controller")])

        self.assertEqual(reply, "Chat 回复")
        self.assertEqual(
            fake_client.chat.completions.request_body["messages"],
            [
                {"role": "system", "content": "系统提示"},
                {"role": "user", "content": "分析 Controller"},
            ],
        )

    def test_generate_reply_uses_google_generate_content_format(self) -> None:
        fake_http_client = _FakeGoogleHttpClient()
        config = _config(format=LLMFormat.GOOGLE_GENERATE_CONTENT, provider="google", base_url=DEFAULT_GOOGLE_BASE_URL)
        llm = RigelLLM(config, http_client=fake_http_client)

        reply = llm.generate_reply([LLMMessage(role="user", content="分析 RepositoryIndexer")])

        self.assertEqual(reply, "Google 回复")
        self.assertEqual(fake_http_client.path, "/models/test-model:generateContent")
        self.assertEqual(fake_http_client.headers["x-goog-api-key"], "test-key")
        self.assertEqual(
            fake_http_client.json_body["contents"],
            [{"role": "user", "parts": [{"text": "分析 RepositoryIndexer"}]}],
        )
        self.assertEqual(fake_http_client.json_body["system_instruction"], {"parts": [{"text": "系统提示"}]})


def _write_llm_config(repository_path: Path, config: dict[str, object]) -> Path:
    config_path = repository_path / ".rigel" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps({"llm": config}, ensure_ascii=False) + "\n", encoding="utf-8")
    return repository_path


def _config(
    *,
    format: LLMFormat,
    provider: str = "openai",
    base_url: str | None = None,
) -> LLMConfig:
    return LLMConfig(
        provider=provider,
        format=format,
        model="test-model",
        api_key="test-key",
        base_url=base_url,
        timeout_seconds=30,
        system_prompt="系统提示",
    )


class _FakeOpenAIClient:
    def __init__(self) -> None:
        self.responses = _FakeResponsesResource()
        self.chat = SimpleNamespace(completions=_FakeChatCompletionsResource())


class _FakeResponsesResource:
    def __init__(self) -> None:
        self.request_body: dict[str, object] = {}

    def create(self, **request_body: object) -> SimpleNamespace:
        self.request_body = request_body
        return SimpleNamespace(output_text="Responses 回复")


class _FakeChatCompletionsResource:
    def __init__(self) -> None:
        self.request_body: dict[str, object] = {}

    def create(self, **request_body: object) -> SimpleNamespace:
        self.request_body = request_body
        message = SimpleNamespace(content="Chat 回复")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _FakeGoogleHttpClient:
    def __init__(self) -> None:
        self.path = ""
        self.headers: dict[str, str] = {}
        self.json_body: dict[str, object] = {}

    def post(self, path: str, *, headers: dict[str, str], json: dict[str, object]) -> "_FakeGoogleResponse":
        self.path = path
        self.headers = headers
        self.json_body = json
        return _FakeGoogleResponse()


class _FakeGoogleResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Google 回复"}],
                    }
                }
            ]
        }
