from __future__ import annotations

from unittest import TestCase
from unittest.mock import patch

from rigel_demo.graphrag.chat import GraphRAGSDKChatService, _litellm_model_name
from rigel_demo.graphrag.config import GraphRAGConfig
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


class _FakeChatSession:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def send_message(self, message: str) -> dict[str, object]:
        self.messages.append(message)
        return {
            "response": "GraphRAG 回复",
            "cypher": "MATCH (entity:Entity) RETURN entity",
            "context": [{"id": "entity:demo:PaymentService"}],
        }


def _llm_config(*, model: str = "gpt-5.2") -> LLMConfig:
    return LLMConfig(
        provider="openai",
        model=model,
        api_key="fake-key",
        base_url=None,
        timeout_seconds=1,
        system_prompt="系统提示",
    )
