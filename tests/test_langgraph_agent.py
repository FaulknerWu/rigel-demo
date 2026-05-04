from __future__ import annotations

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from rigel_demo.agent.langgraph_agent import (
    LangGraphCodeAgent,
    _build_code_graph_tools,
    _chat_model_from_config,
)
from rigel_demo.llm import LLMConfig, LLMMessage


class LangGraphCodeAgentTest(TestCase):
    def test_generate_reply_returns_final_message_and_tool_trace(self) -> None:
        config = _config()
        fake_graph = _FakeCompiledGraph()

        with (
            patch("rigel_demo.agent.langgraph_agent._build_code_graph_tools", return_value=[]),
            patch("rigel_demo.agent.langgraph_agent._chat_model_from_config", return_value=object()),
            patch("rigel_demo.agent.langgraph_agent.create_agent", return_value=fake_graph),
        ):
            agent = LangGraphCodeAgent(
                config=config,
                graph_reader=SimpleNamespace(),
                source_reader=SimpleNamespace(),
                embedding_client=SimpleNamespace(),
            )
            reply = agent.generate_reply([LLMMessage(role="user", content="PaymentService 做什么")])

        self.assertEqual(reply.content, "PaymentService 处理付款流程")
        self.assertEqual(reply.tool_calls[0].name, "recall")
        self.assertEqual(reply.tool_calls[0].args, {"query": "PaymentService"})
        self.assertIsInstance(fake_graph.request["messages"][0], HumanMessage)
        self.assertEqual(fake_graph.config["recursion_limit"], 12)

    def test_chat_model_uses_existing_llm_config_fields(self) -> None:
        config = _config(temperature=0, max_output_tokens=256)

        model = _chat_model_from_config(config)

        self.assertEqual(model.model_name, "test-model")
        self.assertEqual(model.temperature, 0)
        self.assertEqual(model.max_tokens, 256)
        self.assertEqual(model.openai_api_base, "https://example.test/v1")

    def test_tool_business_error_is_returned_as_json(self) -> None:
        tools = _build_code_graph_tools(
            graph_reader=SimpleNamespace(anchors_for_node=lambda _node_id: None),
            source_reader=SimpleNamespace(),
            embedding_client=SimpleNamespace(),
        )
        anchors_tool = next(tool for tool in tools if tool.name == "anchors")

        result = anchors_tool.invoke({"node_id": "entity:missing"})

        self.assertIn('"status": "error"', result)
        self.assertIn("未找到节点", result)


class _FakeCompiledGraph:
    def __init__(self) -> None:
        self.request: dict[str, object] = {}
        self.config: dict[str, object] = {}

    def invoke(self, request: dict[str, object], config: dict[str, object]) -> dict[str, object]:
        self.request = request
        self.config = config
        return {
            "messages": [
                *request["messages"],
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "recall",
                            "args": {"query": "PaymentService"},
                            "id": "tool-call-1",
                        }
                    ],
                ),
                ToolMessage(content='{"status":"success"}', name="recall", tool_call_id="tool-call-1"),
                AIMessage(content="PaymentService 处理付款流程"),
            ]
        }


def _config(
    *,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
) -> LLMConfig:
    return LLMConfig(
        provider="openai",
        model="test-model",
        api_key="test-key",
        base_url="https://example.test/v1",
        timeout_seconds=30,
        system_prompt="系统提示",
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
