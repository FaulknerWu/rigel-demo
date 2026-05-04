from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from falkordb import FalkorDB

from rigel_demo.graphrag.chat import (
    GraphRAGSDKChatService,
    _apply_litellm_environment,
    _litellm_additional_params,
    _litellm_model_name,
    _restore_litellm_environment,
    _start_embedded_falkordb_runtime,
)
from rigel_demo.config import GraphRAGConfig
from rigel_demo.llm import LLMConfig, LLMMessage


class GraphRAGSDKChatServiceTest(TestCase):
    def test_send_messages_uses_knowledge_graph_chat_session(self) -> None:
        fake_chat_session = _FakeChatSession()
        config = _llm_config()

        with patch("rigel_demo.graphrag.chat._build_chat_session", return_value=fake_chat_session):
            chat = GraphRAGSDKChatService(
                config=config,
                graphrag_config=GraphRAGConfig(
                    host="127.0.0.1",
                    port=6379,
                    username=None,
                    password=None,
                ),
                graph_name="rigel",
            )
            reply = chat.send_messages(
                [
                    LLMMessage(role="user", content="PaymentService 做什么"),
                    LLMMessage(role="assistant", content="它处理支付"),
                    LLMMessage(role="user", content="它依赖什么"),
                ]
            )

        self.assertEqual(fake_chat_session.messages, ["PaymentService 做什么", "它依赖什么"])
        self.assertEqual(reply.content, "GraphRAG 回复")
        self.assertEqual(reply.traces[0].name, "cypher")
        self.assertEqual(reply.traces[0].args["query"], "MATCH (entity:Entity) RETURN entity")

    def test_litellm_model_name_adds_provider_prefix(self) -> None:
        self.assertEqual(_litellm_model_name(_llm_config(model="gpt-5.2")), "openai/gpt-5.2")
        self.assertEqual(_litellm_model_name(_llm_config(model="openai/gpt-5.2")), "openai/gpt-5.2")

    def test_litellm_environment_uses_chat_config_api_key(self) -> None:
        config = _llm_config(api_key="config-key", base_url="https://api.example.test/v1")
        original_api_key = os.environ.pop("OPENAI_API_KEY", None)
        original_api_base = os.environ.pop("OPENAI_API_BASE", None)
        try:
            snapshot = _apply_litellm_environment(config)
            self.assertEqual(os.environ["OPENAI_API_KEY"], "config-key")
            self.assertEqual(os.environ["OPENAI_API_BASE"], "https://api.example.test/v1")
            self.assertEqual(_litellm_additional_params(config), {"api_base": "https://api.example.test/v1"})
            _restore_litellm_environment(snapshot)
            self.assertNotIn("OPENAI_API_KEY", os.environ)
            self.assertNotIn("OPENAI_API_BASE", os.environ)
        finally:
            if original_api_key is not None:
                os.environ["OPENAI_API_KEY"] = original_api_key
            if original_api_base is not None:
                os.environ["OPENAI_API_BASE"] = original_api_base

    def test_send_messages_accepts_object_response_text(self) -> None:
        fake_chat_session = _FakeChatSession(response=SimpleNamespace(answer="对象回复"))

        with patch("rigel_demo.graphrag.chat._build_chat_session", return_value=fake_chat_session):
            chat = GraphRAGSDKChatService(
                config=_llm_config(),
                graphrag_config=GraphRAGConfig(
                    host="127.0.0.1",
                    port=6379,
                    username=None,
                    password=None,
                ),
                graph_name="rigel",
            )
            reply = chat.send_messages([LLMMessage(role="user", content="PaymentService 做什么")])

        self.assertEqual(reply.content, "对象回复")
        self.assertEqual(reply.traces, [])

    def test_config_missing_file_keeps_file_not_found_error(self) -> None:
        with TemporaryDirectory() as workspace:
            with self.assertRaisesRegex(FileNotFoundError, "rigel init"):
                GraphRAGConfig.from_repository(Path(workspace))

    def test_embedded_falkordblite_runtime_exposes_repository_database_over_tcp(self) -> None:
        with TemporaryDirectory() as workspace:
            database_path = Path(workspace) / "falkordb.db"
            runtime = _start_embedded_falkordb_runtime(
                database_path=database_path,
                host="127.0.0.1",
            )
            try:
                runtime.client.select_graph("rigel").query("CREATE (:RigelNode {id: 'repo:demo'})")
                rows = FalkorDB(host=runtime.host, port=runtime.port).select_graph("rigel").query(
                    "MATCH (node:RigelNode) RETURN node.id"
                ).result_set
            finally:
                runtime.client.close()

        self.assertEqual(rows, [["repo:demo"]])


class _FakeChatSession:
    def __init__(self, response: object | None = None) -> None:
        self.messages: list[str] = []
        self.response = response

    def send_message(self, message: str) -> object:
        self.messages.append(message)
        if self.response is not None:
            return self.response
        return {
            "response": "GraphRAG 回复",
            "cypher": "MATCH (entity:Entity) RETURN entity",
            "context": [{"id": "entity:demo:PaymentService"}],
        }


def _llm_config(
    *,
    model: str = "gpt-5.2",
    api_key: str = "fake-key",
    base_url: str | None = None,
) -> LLMConfig:
    return LLMConfig(
        provider="openai",
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout_seconds=1,
        system_prompt="系统提示",
    )
