from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase

from falkordb import FalkorDB

from rigel_demo.config import GraphRAGConfig
from rigel_demo.graphrag.chat import (
    LangGraphChatService,
    RigelGraphRAGError,
    _apply_safe_limit,
    _is_safe_readonly_cypher,
    _sanitize_generated_cypher,
    _start_embedded_falkordb_runtime,
)
from rigel_demo.llm import LLMConfig, LLMMessage


class LangGraphChatServiceTest(TestCase):
    def test_send_messages_generates_cypher_executes_query_and_answers(self) -> None:
        fake_model = _FakeChatModel(["MATCH (entity:Entity) RETURN entity", "PaymentService 处理支付流程"])
        fake_graph = _FakeGraph()
        chat = LangGraphChatService(
            config=_llm_config(),
            graphrag_config=_graphrag_config(),
            graph_name="rigel",
            chat_model=fake_model,
            graph=fake_graph,
        )

        reply = chat.send_messages([LLMMessage(role="user", content="PaymentService 做什么")])

        self.assertEqual(reply.content, "PaymentService 处理支付流程")
        self.assertEqual(fake_graph.queries, ["MATCH (entity:Entity) RETURN entity LIMIT 20"])
        self.assertEqual([trace.name for trace in reply.traces], ["cypher", "context"])
        self.assertEqual(reply.traces[0].args["query"], "MATCH (entity:Entity) RETURN entity LIMIT 20")
        self.assertEqual(reply.traces[1].args["items"], 1)

    def test_send_messages_passes_recent_assistant_answer_to_followup_prompt(self) -> None:
        fake_model = _FakeChatModel(["MATCH (entity:Entity) RETURN entity LIMIT 5", "它依赖 Repository"])
        chat = LangGraphChatService(
            config=_llm_config(),
            graphrag_config=_graphrag_config(),
            graph_name="rigel",
            chat_model=fake_model,
            graph=_FakeGraph(),
        )

        chat.send_messages(
            [
                LLMMessage(role="user", content="PaymentService 做什么"),
                LLMMessage(role="assistant", content="它处理付款"),
                LLMMessage(role="user", content="它依赖什么"),
            ]
        )

        cypher_prompt = fake_model.calls[0][1][1]
        self.assertIn("上一轮回答：它处理付款", cypher_prompt)
        self.assertIn("用户追问：它依赖什么", cypher_prompt)

    def test_send_messages_rejects_empty_messages(self) -> None:
        chat = LangGraphChatService(
            config=_llm_config(),
            graphrag_config=_graphrag_config(),
            graph_name="rigel",
            chat_model=_FakeChatModel([]),
            graph=_FakeGraph(),
        )

        with self.assertRaisesRegex(RigelGraphRAGError, "消息列表不能为空"):
            chat.send_messages([])

    def test_send_messages_rejects_last_non_user_message(self) -> None:
        chat = LangGraphChatService(
            config=_llm_config(),
            graphrag_config=_graphrag_config(),
            graph_name="rigel",
            chat_model=_FakeChatModel([]),
            graph=_FakeGraph(),
        )

        with self.assertRaisesRegex(RigelGraphRAGError, "最后一条消息必须来自用户"):
            chat.send_messages([LLMMessage(role="assistant", content="回复")])

    def test_unsafe_cypher_is_rejected_without_query_execution(self) -> None:
        fake_graph = _FakeGraph()
        chat = LangGraphChatService(
            config=_llm_config(),
            graphrag_config=_graphrag_config(),
            graph_name="rigel",
            chat_model=_FakeChatModel(["MATCH (n) DELETE n", "不应调用"]),
            graph=fake_graph,
        )

        reply = chat.send_messages([LLMMessage(role="user", content="删除所有节点")])

        self.assertEqual(reply.content, "无法生成安全只读查询，因此没有执行图数据库查询。")
        self.assertEqual(fake_graph.queries, [])
        self.assertEqual(reply.traces[0].args["rejected"], True)

    def test_regex_match_is_rewritten_to_falkordb_contains(self) -> None:
        fake_model = _FakeChatModel(
            [
                "MATCH (entity:Entity) WHERE entity.display_name =~ '(?i).*PaymentService.*' RETURN entity",
                "PaymentService 处理支付流程",
            ]
        )
        fake_graph = _FakeGraph()
        chat = LangGraphChatService(
            config=_llm_config(),
            graphrag_config=_graphrag_config(),
            graph_name="rigel",
            chat_model=fake_model,
            graph=fake_graph,
        )

        reply = chat.send_messages([LLMMessage(role="user", content="PaymentService 做什么")])

        self.assertEqual(reply.content, "PaymentService 处理支付流程")
        self.assertEqual(
            fake_graph.queries,
            [
                "MATCH (entity:Entity) WHERE toLower(coalesce(entity.display_name, '')) "
                "CONTAINS 'paymentservice' RETURN entity LIMIT 20"
            ],
        )

    def test_complex_regex_match_is_rejected_without_query_execution(self) -> None:
        fake_model = _FakeChatModel(["MATCH (entity:Entity) WHERE entity.display_name =~ 'Pay.*Service|Order' RETURN entity"])
        fake_graph = _FakeGraph()
        chat = LangGraphChatService(
            config=_llm_config(),
            graphrag_config=_graphrag_config(),
            graph_name="rigel",
            chat_model=fake_model,
            graph=fake_graph,
        )

        reply = chat.send_messages([LLMMessage(role="user", content="PaymentService 做什么")])

        self.assertEqual(reply.content, "无法生成安全只读查询，因此没有执行图数据库查询。")
        self.assertEqual(fake_graph.queries, [])
        self.assertEqual(reply.traces[0].args["rejected"], True)

    def test_query_execution_failure_degrades_to_empty_context_answer(self) -> None:
        fake_model = _FakeChatModel(["MATCH (entity:Entity) RETURN entity", "无法从当前图谱确认"])
        fake_graph = _FakeGraph(error=RuntimeError("Generated Cypher Statement is not valid"))
        chat = LangGraphChatService(
            config=_llm_config(),
            graphrag_config=_graphrag_config(),
            graph_name="rigel",
            chat_model=fake_model,
            graph=fake_graph,
        )

        reply = chat.send_messages([LLMMessage(role="user", content="PaymentService 做什么")])

        self.assertEqual(reply.content, "无法从当前图谱确认")
        self.assertEqual(reply.traces[1].args["items"], 0)
        self.assertIn("图查询执行失败", str(reply.traces[0].args["error"]))

    def test_safe_cypher_validation_and_limit(self) -> None:
        self.assertTrue(_is_safe_readonly_cypher("MATCH (n) RETURN n"))
        self.assertTrue(
            _is_safe_readonly_cypher(
                "MATCH (n) WHERE toLower(coalesce(n.name, '')) CONTAINS 'payment' RETURN n"
            )
        )
        self.assertFalse(_is_safe_readonly_cypher("CREATE (:Entity)"))
        self.assertFalse(_is_safe_readonly_cypher("MATCH (n) RETURN n; MATCH (m) RETURN m"))
        self.assertFalse(_is_safe_readonly_cypher("CALL db.labels()"))
        self.assertFalse(_is_safe_readonly_cypher("MATCH (n) WHERE n.name =~ '.*Payment.*' RETURN n"))
        self.assertEqual(_apply_safe_limit("MATCH (n) RETURN n"), "MATCH (n) RETURN n LIMIT 20")
        self.assertEqual(_apply_safe_limit("MATCH (n) RETURN n LIMIT 3"), "MATCH (n) RETURN n LIMIT 3")
        self.assertEqual(
            _sanitize_generated_cypher("MATCH (n) WHERE n.name =~ '(?i).*Payment.*' RETURN n"),
            "MATCH (n) WHERE toLower(coalesce(n.name, '')) CONTAINS 'payment' RETURN n",
        )

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


class _FakeChatModel:
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.calls: list[list[tuple[str, str]]] = []

    def invoke(self, messages: list[tuple[str, str]]) -> SimpleNamespace:
        self.calls.append(messages)
        return SimpleNamespace(content=self._responses.pop(0))


class _FakeGraph:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.get_schema = "(:Entity {id: STRING})"
        self.queries: list[str] = []
        self.error = error

    def query(self, cypher: str) -> list[dict[str, object]]:
        self.queries.append(cypher)
        if self.error is not None:
            raise self.error
        return [{"entity": {"id": "entity:demo:PaymentService", "display_name": "PaymentService"}}]


def _llm_config() -> LLMConfig:
    return LLMConfig(
        provider="openai",
        model="gpt-5.2",
        api_key="fake-key",
        base_url=None,
        timeout_seconds=1,
        system_prompt="系统提示",
    )


def _graphrag_config() -> GraphRAGConfig:
    return GraphRAGConfig(
        host="127.0.0.1",
        port=6379,
        username=None,
        password=None,
    )
