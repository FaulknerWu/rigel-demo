from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from rigel_demo.llm import (
    DEFAULT_CHAT_SYSTEM_PROMPT,
    DEFAULT_SUMMARY_SYSTEM_PROMPT,
    LLMConfig,
    LLMConfigSection,
    LLMConfigurationError,
    LLMMessage,
    LangChainSummaryClient,
    build_langchain_chat_model,
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
                            "concurrent_requests": 6,
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
        self.assertEqual(config.concurrent_requests, 6)
        self.assertIn("摘要生成器", config.system_prompt)
        self.assertIn("职责", config.system_prompt)
        self.assertIn("检索别名", config.system_prompt)

    def test_summary_section_rejects_invalid_concurrent_requests(self) -> None:
        with TemporaryDirectory() as workspace:
            repository_path = _write_llm_config(
                Path(workspace),
                {
                    "provider": "openai",
                    "model": "gpt-5.2-mini",
                    "api_key": "summary-key",
                    "base_url": None,
                    "timeout_seconds": 60,
                    "temperature": 0,
                    "max_output_tokens": 256,
                    "concurrent_requests": 0,
                    "system_prompt": DEFAULT_SUMMARY_SYSTEM_PROMPT,
                },
                section=LLMConfigSection.SUMMARY,
            )

            with self.assertRaisesRegex(LLMConfigurationError, "summary.concurrent_requests"):
                LLMConfig.from_repository(repository_path, LLMConfigSection.SUMMARY)

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


class LangChainLLMTest(TestCase):
    def test_build_langchain_chat_model_uses_project_config(self) -> None:
        config = _config(temperature=0, max_output_tokens=300)

        with patch("langchain_openai.ChatOpenAI", _FakeChatOpenAI):
            chat_model = build_langchain_chat_model(config)

        self.assertEqual(
            chat_model.kwargs,
            {
                "model": "test-model",
                "api_key": "test-key",
                "timeout": 30,
                "temperature": 0,
                "model_kwargs": {"max_completion_tokens": 300},
            },
        )

    def test_summary_client_invokes_chatopenai_with_system_prompt(self) -> None:
        fake_model = _FakeChatModel()
        summary_client = LangChainSummaryClient(_config(), chat_model=fake_model)

        reply = summary_client.generate_reply([LLMMessage(role="user", content="分析 Controller")])

        self.assertEqual(reply, "Chat 回复")
        self.assertEqual(
            fake_model.messages,
            [
                ("system", "系统提示"),
                ("user", "分析 Controller"),
            ],
        )

    def test_summary_client_rejects_blank_message_content(self) -> None:
        fake_model = _FakeChatModel()
        summary_client = LangChainSummaryClient(_config(), chat_model=fake_model)

        with self.assertRaisesRegex(ValueError, "消息内容不能为空"):
            summary_client.generate_reply([LLMMessage(role="user", content="   ")])

        self.assertEqual(fake_model.messages, [])

    def test_generate_reply_strips_messages_before_request(self) -> None:
        fake_model = _FakeChatModel()
        summary_client = LangChainSummaryClient(_config(), chat_model=fake_model)

        reply = summary_client.generate_reply([LLMMessage(role="user", content="  分析 Controller  ")])

        self.assertEqual(reply, "Chat 回复")
        self.assertEqual(
            fake_model.messages,
            [
                ("system", "系统提示"),
                ("user", "分析 Controller"),
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


class _FakeChatOpenAI:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class _FakeChatModel:
    def __init__(self, response: str = "Chat 回复") -> None:
        self.messages: list[tuple[str, str]] = []
        self.response = response

    def invoke(self, messages: list[tuple[str, str]]) -> SimpleNamespace:
        self.messages = messages
        return SimpleNamespace(content=self.response)
