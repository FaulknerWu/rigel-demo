from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from fastapi.testclient import TestClient

from rigel_demo.llm import LLMConfig, LLMFormat, LLMMessage
from rigel_demo.web.app import create_app


class WebAppLLMTest(TestCase):
    def test_chat_returns_configuration_error_when_llm_is_missing(self) -> None:
        with TemporaryDirectory() as workspace:
            client = TestClient(create_app(Path(workspace)))

            response = client.post("/api/chat", json={"messages": [{"role": "user", "content": "你好"}]})

        self.assertEqual(response.status_code, 503)
        self.assertIn(".rigel/config.json", response.json()["detail"])

    def test_chat_uses_injected_llm_client(self) -> None:
        fake_llm = _FakeRigelLLM()
        with TemporaryDirectory() as workspace:
            client = TestClient(create_app(Path(workspace), llm_client=fake_llm))

            response = client.post("/api/chat", json={"messages": [{"role": "user", "content": "你好"}]})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["message"], {"role": "assistant", "content": "测试回复"})
        self.assertEqual(fake_llm.messages, [LLMMessage(role="user", content="你好")])


class _FakeRigelLLM:
    def __init__(self) -> None:
        self.config = LLMConfig(
            provider="openai",
            format=LLMFormat.OPENAI_CHAT,
            model="fake-model",
            api_key="fake-key",
            base_url=None,
            timeout_seconds=1,
            system_prompt="系统提示",
        )
        self.messages: list[LLMMessage] = []

    def generate_reply(self, messages: list[LLMMessage]) -> str:
        self.messages = messages
        return "测试回复"
