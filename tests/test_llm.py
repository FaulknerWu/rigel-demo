from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase

from rigel_demo.llm import (
    DEFAULT_CHAT_SYSTEM_PROMPT,
    DEFAULT_SUMMARY_SYSTEM_PROMPT,
    LLMConfig,
    LLMConfigSection,
    LLMConfigurationError,
    LLMMessage,
    LLMResponseError,
    RigelLLM,
)


class LLMConfigTest(TestCase):
    def test_openai_provider_uses_chat_completions_defaults(self) -> None:
        with TemporaryDirectory() as workspace:
            repository_path = _write_llm_config(
                Path(workspace),
                {
                    "provider": "openai",
                    "model": "gpt-5.2",
                    "api_key": "openai-key",
                    "base_url": None,
                    "timeout_seconds": 60,
                    "temperature": None,
                    "max_output_tokens": None,
                    "system_prompt": DEFAULT_CHAT_SYSTEM_PROMPT,
                },
            )

            config = LLMConfig.from_repository(repository_path)

        self.assertEqual(config.provider, "openai")
        self.assertEqual(config.section, LLMConfigSection.CHAT)
        self.assertEqual(config.model, "gpt-5.2")
        self.assertEqual(config.api_key, "openai-key")
        self.assertIsNone(config.base_url)
        self.assertEqual(config.timeout_seconds, 60)
        self.assertEqual(config.system_prompt, DEFAULT_CHAT_SYSTEM_PROMPT)

    def test_summary_section_reads_independent_model(self) -> None:
        with TemporaryDirectory() as workspace:
            repository_path = Path(workspace)
            config_path = repository_path / ".rigel" / "config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                json.dumps(
                    {
                        "chat": {
                            "provider": "openai",
                            "model": "gpt-5.2",
                            "api_key": "chat-key",
                            "base_url": None,
                            "timeout_seconds": 60,
                            "temperature": None,
                            "max_output_tokens": None,
                            "system_prompt": DEFAULT_CHAT_SYSTEM_PROMPT,
                        },
                        "summary": {
                            "provider": "openai",
                            "model": "gpt-5.2-mini",
                            "api_key": "summary-key",
                            "base_url": None,
                            "timeout_seconds": 60,
                            "temperature": 0,
                            "max_output_tokens": 256,
                            "system_prompt": DEFAULT_SUMMARY_SYSTEM_PROMPT,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            config = LLMConfig.from_repository(repository_path, LLMConfigSection.SUMMARY)

        self.assertEqual(config.section, LLMConfigSection.SUMMARY)
        self.assertEqual(config.model, "gpt-5.2-mini")
        self.assertEqual(config.api_key, "summary-key")
        self.assertEqual(config.max_output_tokens, 256)
        self.assertIn("摘要生成器", config.system_prompt)

    def test_chat_completions_config_supports_custom_base_url(self) -> None:
        with TemporaryDirectory() as workspace:
            repository_path = _write_llm_config(
                Path(workspace),
                {
                    "provider": "openai",
                    "model": "gpt-custom",
                    "api_key": "openai-key",
                    "base_url": "https://example.test/v1",
                    "timeout_seconds": 60,
                    "temperature": None,
                    "max_output_tokens": 1024,
                    "system_prompt": DEFAULT_CHAT_SYSTEM_PROMPT,
                },
            )

            config = LLMConfig.from_repository(repository_path)

        self.assertEqual(config.provider, "openai")
        self.assertEqual(config.base_url, "https://example.test/v1")
        self.assertEqual(config.max_output_tokens, 1024)

    def test_custom_provider_requires_base_url(self) -> None:
        with TemporaryDirectory() as workspace:
            repository_path = _write_llm_config(
                Path(workspace),
                {
                    "provider": "acme",
                    "model": "acme-chat",
                    "api_key": "acme-key",
                    "base_url": None,
                    "timeout_seconds": 60,
                    "temperature": None,
                    "max_output_tokens": None,
                    "system_prompt": DEFAULT_CHAT_SYSTEM_PROMPT,
                },
            )

            with self.assertRaisesRegex(LLMConfigurationError, "chat.base_url"):
                LLMConfig.from_repository(repository_path)

    def test_custom_provider_accepts_base_url(self) -> None:
        with TemporaryDirectory() as workspace:
            repository_path = _write_llm_config(
                Path(workspace),
                {
                    "provider": "acme",
                    "model": "acme-chat",
                    "api_key": "acme-key",
                    "base_url": "https://acme.example/v1",
                    "timeout_seconds": 60,
                    "temperature": None,
                    "max_output_tokens": None,
                    "system_prompt": DEFAULT_CHAT_SYSTEM_PROMPT,
                },
            )

            config = LLMConfig.from_repository(repository_path)

        self.assertEqual(config.provider, "acme")
        self.assertEqual(config.api_key, "acme-key")
        self.assertEqual(config.base_url, "https://acme.example/v1")

    def test_missing_model_fails_fast(self) -> None:
        with TemporaryDirectory() as workspace:
            repository_path = _write_llm_config(
                Path(workspace),
                {
                    "provider": "openai",
                    "api_key": "test-key",
                    "base_url": None,
                    "timeout_seconds": 60,
                    "temperature": None,
                    "max_output_tokens": None,
                    "system_prompt": DEFAULT_CHAT_SYSTEM_PROMPT,
                },
            )

            with self.assertRaisesRegex(LLMConfigurationError, "chat.model"):
                LLMConfig.from_repository(repository_path)

    def test_missing_template_field_fails_fast(self) -> None:
        with TemporaryDirectory() as workspace:
            repository_path = _write_llm_config(
                Path(workspace),
                {
                    "provider": "openai",
                    "model": "gpt-5.2",
                    "api_key": "test-key",
                    "base_url": None,
                    "temperature": None,
                    "max_output_tokens": None,
                    "system_prompt": DEFAULT_CHAT_SYSTEM_PROMPT,
                },
            )

            with self.assertRaisesRegex(LLMConfigurationError, "chat.timeout_seconds"):
                LLMConfig.from_repository(repository_path)

    def test_missing_config_file_fails_fast(self) -> None:
        with TemporaryDirectory() as workspace:
            with self.assertRaisesRegex(LLMConfigurationError, ".rigel/config.json"):
                LLMConfig.from_repository(Path(workspace))


class RigelLLMTest(TestCase):
    def test_generate_reply_uses_chat_completions_request_shape(self) -> None:
        fake_client = _FakeOpenAIClient()
        config = _config(temperature=0, max_output_tokens=300)
        llm = RigelLLM(config, openai_client=fake_client)

        reply = llm.generate_reply([LLMMessage(role="user", content="分析 Controller")])

        self.assertEqual(reply, "Chat 回复")
        self.assertEqual(
            fake_client.chat.completions.request_body,
            {
                "model": "test-model",
                "messages": [
                    {"role": "system", "content": "系统提示"},
                    {"role": "user", "content": "分析 Controller"},
                ],
                "temperature": 0,
                "max_completion_tokens": 300,
            },
        )

    def test_generate_reply_rejects_blank_message_content(self) -> None:
        fake_client = _FakeOpenAIClient()
        llm = RigelLLM(_config(), openai_client=fake_client)

        with self.assertRaisesRegex(LLMResponseError, "消息内容不能为空"):
            llm.generate_reply([LLMMessage(role="user", content="   ")])

        self.assertEqual(fake_client.chat.completions.request_body, {})

    def test_generate_reply_strips_messages_before_request(self) -> None:
        fake_client = _FakeOpenAIClient()
        llm = RigelLLM(_config(), openai_client=fake_client)

        reply = llm.generate_reply([LLMMessage(role="user", content="  分析 Controller  ")])

        self.assertEqual(reply, "Chat 回复")
        self.assertEqual(
            fake_client.chat.completions.request_body["messages"],
            [
                {"role": "system", "content": "系统提示"},
                {"role": "user", "content": "分析 Controller"},
            ],
        )


def _write_llm_config(
    repository_path: Path,
    config: dict[str, object],
    *,
    section: LLMConfigSection = LLMConfigSection.CHAT,
) -> Path:
    config_path = repository_path / ".rigel" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps({section.value: config}, ensure_ascii=False) + "\n", encoding="utf-8")
    return repository_path


def _config(
    *,
    provider: str = "openai",
    base_url: str | None = None,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
) -> LLMConfig:
    return LLMConfig(
        provider=provider,
        model="test-model",
        api_key="test-key",
        base_url=base_url,
        timeout_seconds=30,
        system_prompt="系统提示",
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )


class _FakeOpenAIClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_FakeChatCompletionsResource())


class _FakeChatCompletionsResource:
    def __init__(self) -> None:
        self.request_body: dict[str, object] = {}

    def create(self, **request_body: object) -> SimpleNamespace:
        self.request_body = request_body
        message = SimpleNamespace(content="Chat 回复")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])
