from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase

from falkordb import FalkorDB

from rigel_demo.config import GraphRAGConfig
from rigel_demo.embedding import EmbeddingConfig, EmbeddingFormat, EmbeddingInputMode
from rigel_demo.graphrag.chat import (
    LangGraphChatService,
    RigelGraphRAGError,
    _start_embedded_falkordb_runtime,
)
from rigel_demo.llm import LLMConfig, LLMMessage


class LangGraphChatServiceTest(TestCase):
    def test_send_messages_uses_vector_seed_tool_and_passes_evidence_to_answer(self) -> None:
        fake_model = _FakeChatModel(
            [
                _tool_call("call:seed", "vector_search_seeds", {"query_text": "PaymentService", "limit": 99}),
                _ai_text("证据足够"),
                _ai_text("PaymentService 处理支付流程"),
            ]
        )
        fake_graph = _FakeGraph()
        chat = LangGraphChatService(
            config=_llm_config(),
            graphrag_config=_graphrag_config(),
            graph_name="rigel",
            embedding_client=_FakeEmbeddingClient(),
            chat_model=fake_model,
            graph=fake_graph,
        )

        reply = chat.send_messages([LLMMessage(role="user", content="PaymentService 做什么")])

        self.assertEqual(reply.content, "PaymentService 处理支付流程")
        self.assertEqual(reply.traces[0].name, "vector_search_seeds")
        self.assertEqual(reply.traces[0].args["query_text"], "PaymentService")
        self.assertIn("limit 超过上限 5，已裁剪", reply.traces[0].args["warnings"])
        answer_prompt = fake_model.calls[-1][1][1]
        self.assertIn("PaymentService 处理付款流程", answer_prompt)
        self.assertIn("entity:demo:PaymentService", answer_prompt)
        self.assertEqual(fake_graph.queries[0]["params"]["vector_limit"], 5)

    def test_expand_neighbors_allows_second_hop_and_skips_visited_node(self) -> None:
        fake_model = _FakeChatModel(
            [
                _tool_call("call:seed", "vector_search_seeds", {"query_text": "PaymentService"}),
                _tool_call("call:first-hop", "expand_neighbors", {"node_ids": ["entity:demo:PaymentService"]}),
                _tool_call(
                    "call:second-hop",
                    "expand_neighbors",
                    {"node_ids": ["entity:demo:PaymentService", "file:demo:PaymentService.java"]},
                ),
                _ai_text("证据足够"),
                _ai_text("PaymentService 位于 PaymentService.java"),
            ]
        )
        chat = LangGraphChatService(
            config=_llm_config(),
            graphrag_config=_graphrag_config(),
            graph_name="rigel",
            embedding_client=_FakeEmbeddingClient(),
            chat_model=fake_model,
            graph=_FakeGraph(),
        )

        reply = chat.send_messages([LLMMessage(role="user", content="PaymentService 在哪里")])

        expand_traces = [trace for trace in reply.traces if trace.name == "expand_neighbors"]
        self.assertEqual(len(expand_traces), 2)
        self.assertEqual(expand_traces[0].args["relations"][0]["node"]["id"], "file:demo:PaymentService.java")
        self.assertEqual(expand_traces[1].args["relations"][0]["node"]["id"], "module:demo:root")
        self.assertIn("已扩展节点已忽略：entity:demo:PaymentService", expand_traces[1].args["warnings"])

    def test_query_relation_supports_direction_and_visible_relation_whitelist(self) -> None:
        chat = LangGraphChatService(
            config=_llm_config(),
            graphrag_config=_graphrag_config(),
            graph_name="rigel",
            embedding_client=_FakeEmbeddingClient(),
            chat_model=_FakeChatModel([]),
            graph=_FakeGraph(),
        )
        known_node_ids = {"entity:demo:PaymentService"}

        outgoing = chat._query_relation(
            {
                "node_ids": ["entity:demo:PaymentService"],
                "relation_type": "DEPENDS_ON",
                "direction": "outgoing",
            },
            known_node_ids=known_node_ids,
        )
        incoming = chat._query_relation(
            {
                "node_ids": ["entity:demo:PaymentService"],
                "relation_type": "DEPENDS_ON",
                "direction": "incoming",
            },
            known_node_ids=known_node_ids,
        )
        both = chat._query_relation(
            {
                "node_ids": ["entity:demo:PaymentService"],
                "relation_type": "DESCRIBES",
                "direction": "both",
            },
            known_node_ids=known_node_ids,
        )

        self.assertEqual(outgoing["items"][0]["direction"], "outgoing")
        self.assertEqual(incoming["items"][0]["direction"], "incoming")
        self.assertEqual(both["items"], [])
        self.assertIn("非法关系类型已忽略：DESCRIBES", both["warnings"])

    def test_tool_security_warnings_for_unknown_node_illegal_edge_and_large_limit(self) -> None:
        fake_model = _FakeChatModel(
            [
                _tool_call("call:seed", "vector_search_seeds", {"query_text": "PaymentService"}),
                _tool_call(
                    "call:expand",
                    "expand_neighbors",
                    {
                        "node_ids": ["entity:demo:PaymentService", "entity:demo:Unknown"],
                        "edge_types": ["CONTAINS", "HAS_ANCHOR"],
                        "limit_per_node": 100,
                    },
                ),
                _ai_text("证据足够"),
                _ai_text("只返回可见关系"),
            ]
        )
        chat = LangGraphChatService(
            config=_llm_config(),
            graphrag_config=_graphrag_config(),
            graph_name="rigel",
            embedding_client=_FakeEmbeddingClient(),
            chat_model=fake_model,
            graph=_FakeGraph(),
        )

        reply = chat.send_messages([LLMMessage(role="user", content="PaymentService 的关系")])

        expand_trace = next(trace for trace in reply.traces if trace.name == "expand_neighbors")
        self.assertIn("未知节点已忽略：entity:demo:Unknown", expand_trace.args["warnings"])
        self.assertIn("非法边类型已忽略：HAS_ANCHOR", expand_trace.args["warnings"])
        self.assertIn("limit_per_node 超过上限 8，已裁剪", expand_trace.args["warnings"])

    def test_tool_call_limit_stops_retrieval_and_generates_answer(self) -> None:
        fake_model = _FakeChatModel(
            [
                _tool_call("call:seed", "vector_search_seeds", {"query_text": "PaymentService"}),
                _ai_tool_calls(
                    [
                        {
                            "id": f"call:expand:{index}",
                            "name": "expand_neighbors",
                            "args": {"node_ids": ["entity:demo:PaymentService"]},
                        }
                        for index in range(4)
                    ]
                ),
                _ai_text("无法从当前图谱确认更多证据"),
            ]
        )
        chat = LangGraphChatService(
            config=_llm_config(),
            graphrag_config=_graphrag_config(),
            graph_name="rigel",
            embedding_client=_FakeEmbeddingClient(),
            chat_model=fake_model,
            graph=_FakeGraph(),
        )

        reply = chat.send_messages([LLMMessage(role="user", content="继续查")])

        self.assertEqual(reply.content, "无法从当前图谱确认更多证据")
        self.assertTrue(any("工具调用次数达到上限" in str(trace.args) for trace in reply.traces))
        self.assertEqual(len(fake_model.calls), 3)

    def test_send_messages_rejects_empty_messages(self) -> None:
        chat = LangGraphChatService(
            config=_llm_config(),
            graphrag_config=_graphrag_config(),
            graph_name="rigel",
            embedding_client=_FakeEmbeddingClient(),
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
            embedding_client=_FakeEmbeddingClient(),
            chat_model=_FakeChatModel([]),
            graph=_FakeGraph(),
        )

        with self.assertRaisesRegex(RigelGraphRAGError, "最后一条消息必须来自用户"):
            chat.send_messages([LLMMessage(role="assistant", content="回复")])

    def test_send_messages_passes_recent_assistant_answer_to_final_prompt(self) -> None:
        fake_model = _FakeChatModel(
            [
                _tool_call("call:seed", "vector_search_seeds", {"query_text": "PaymentService"}),
                _ai_text("证据足够"),
                _ai_text("它依赖 Repository"),
            ]
        )
        chat = LangGraphChatService(
            config=_llm_config(),
            graphrag_config=_graphrag_config(),
            graph_name="rigel",
            embedding_client=_FakeEmbeddingClient(),
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

        answer_prompt = fake_model.calls[-1][1][1]
        self.assertIn("上一轮回答：\n它处理付款", answer_prompt)

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


def _tool_call(tool_call_id: str, name: str, args: dict[str, object]) -> SimpleNamespace:
    return _ai_tool_calls([{"id": tool_call_id, "name": name, "args": args}])


def _ai_tool_calls(tool_calls: list[dict[str, object]]) -> SimpleNamespace:
    return SimpleNamespace(content="", tool_calls=tool_calls)


def _ai_text(content: str) -> SimpleNamespace:
    return SimpleNamespace(content=content, tool_calls=[])


class _FakeChatModel:
    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self._responses = responses
        self.calls: list[list[object]] = []
        self.bound_tools: list[object] = []

    def bind_tools(self, tools: list[object]) -> "_FakeChatModel":
        self.bound_tools = tools
        return self

    def invoke(self, messages: list[object]) -> SimpleNamespace:
        self.calls.append(messages)
        return self._responses.pop(0)


class _FakeGraph:
    def __init__(self) -> None:
        self.get_schema = "(:Entity {id: STRING})"
        self.queries: list[dict[str, object]] = []

    def query(self, query_text: str, params: dict[str, object]) -> list[dict[str, object]]:
        self.queries.append({"query": query_text, "params": params})
        if "queryNodes" in query_text:
            return [
                {
                    "summary_id": "summary:entity:demo:PaymentService",
                    "summary_properties": {
                        "rigel_type": "Summary",
                        "text": "PaymentService 处理付款流程",
                        "summary_model": "summary-model",
                        "embedding_model": "text-embedding-3-small",
                        "embedding_dimensions": 3,
                        "source_hash": "sha256:payment-service",
                    },
                    "node_id": "entity:demo:PaymentService",
                    "node_properties": _payment_entity_properties(),
                    "distance": 0.1,
                }
            ]

        node_id = str(params["node_id"])
        if "WHERE source.id = $node_id" in query_text:
            return [_relation_row(node_id, "entity:demo:OrderRepository", "DEPENDS_ON")]
        if "WHERE target.id = $node_id" in query_text:
            return [_relation_row("file:demo:PaymentService.java", node_id, "CONTAINS")]
        if node_id == "entity:demo:PaymentService":
            return [_relation_row("file:demo:PaymentService.java", node_id, "CONTAINS")]
        if node_id == "file:demo:PaymentService.java":
            return [_relation_row("module:demo:root", node_id, "CONTAINS")]
        return []


def _relation_row(source_id: str, target_id: str, edge_type: str) -> dict[str, object]:
    return {
        "source_id": source_id,
        "source_properties": _node_properties(source_id),
        "target_id": target_id,
        "target_properties": _node_properties(target_id),
        "edge_type": edge_type,
        "edge_properties": {
            "id": f"{edge_type.lower()}:{source_id}:{target_id}",
            "kind": "test",
        },
    }


def _node_properties(node_id: str) -> dict[str, object]:
    if node_id == "entity:demo:PaymentService":
        return _payment_entity_properties()
    if node_id == "entity:demo:OrderRepository":
        return {
            "rigel_type": "Entity",
            "display_name": "OrderRepository",
            "qualified_name": "demo.OrderRepository",
        }
    if node_id == "file:demo:PaymentService.java":
        return {
            "rigel_type": "File",
            "relative_path": "src/main/java/demo/PaymentService.java",
            "language": "java",
            "zone": "prod",
            "content_hash": "sha256:payment-file",
        }
    return {
        "rigel_type": "Module",
        "name": "root",
        "root_path": ".",
        "ecosystem": "maven",
        "zone": "prod",
    }


def _payment_entity_properties() -> dict[str, object]:
    return {
        "rigel_type": "Entity",
        "display_name": "PaymentService",
        "qualified_name": "demo.PaymentService",
        "kind_norm": "class",
        "kind_raw": "class_declaration",
        "origin": "internal",
        "semantic_hash": "sha256:payment-service",
    }


class _FakeEmbeddingClient:
    def __init__(self) -> None:
        self.config = EmbeddingConfig(
            provider="openai",
            format=EmbeddingFormat.OPENAI_EMBEDDINGS,
            model="text-embedding-3-small",
            api_key="fake-key",
            base_url=None,
            dimensions=3,
            timeout_seconds=1,
            batch_size=8,
            input_mode=EmbeddingInputMode.ARRAY,
        )

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


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
