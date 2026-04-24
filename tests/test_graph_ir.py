import json
import unittest

from rigel_demo import (
    Anchor,
    EdgeType,
    Entity,
    File,
    GraphEdge,
    GraphIR,
    Module,
    NodeType,
    Repository,
    Summary,
)


class GraphIRTest(unittest.TestCase):
    def test_graph_ir_can_be_serialized_to_json(self) -> None:
        graph = GraphIR()
        graph.add_node(Repository(repo_id="repo:rigel", name="rigel"))
        graph.add_node(
            Module(
                module_id="module:core",
                name="core",
                root_path="src/core",
                ecosystem="maven",
                zone="prod",
            )
        )
        graph.add_node(
            File(
                file_id="file:core/UserService.java",
                relative_path="src/core/UserService.java",
                language="java",
                zone="prod",
                content_hash="sha256:file",
            )
        )
        graph.add_node(
            Entity(
                entity_id="entity:UserService",
                entity_key="java:com.example.UserService",
                display_name="UserService",
                qualified_name="com.example.UserService",
                kind_norm="class",
                kind_raw="class_declaration",
                origin="internal",
                semantic_hash="sha256:entity",
            )
        )
        graph.add_node(
            Anchor(
                anchor_id="anchor:UserService:definition",
                start_line=1,
                start_col=1,
                end_line=20,
                end_col=2,
                role="definition",
            )
        )
        graph.add_node(
            Summary(
                summary_id="summary:UserService:retrieval",
                text="用户服务负责用户资料读写。",
                purpose="retrieval",
                source_hash="sha256:entity",
                embedding_model="text-embedding-model",
                embedding=[0.1, 0.2, 0.3],
            )
        )

        graph.add_edge(
            GraphEdge.create(
                EdgeType.CONTAINS,
                "repo:rigel",
                "module:core",
                kind="physical-membership",
                provenance="tree-sitter",
                confidence=1.0,
            )
        )
        graph.add_edge(
            GraphEdge.create(
                EdgeType.DEPENDS_ON,
                "entity:UserService",
                "entity:UserRepository",
                kind="calls",
                provenance="lsp",
                confidence=0.98,
            )
        )
        graph.add_edge(
            GraphEdge.create(
                EdgeType.SPECIALIZES,
                "entity:UserService",
                "entity:BaseService",
                kind="extends",
                provenance="lsp",
                confidence=1.0,
            )
        )
        graph.add_edge(
            GraphEdge.create(
                EdgeType.ALIASES,
                "entity:UserServiceAlias",
                "entity:UserService",
                kind="import-alias",
                provenance="tree-sitter",
                confidence=0.9,
            )
        )
        graph.add_edge(
            GraphEdge.create(
                EdgeType.HAS_ANCHOR,
                "entity:UserService",
                "anchor:UserService:definition",
                role="definition",
                provenance="tree-sitter",
                confidence=1.0,
            )
        )
        graph.add_edge(
            GraphEdge.create(
                EdgeType.DESCRIBES,
                "summary:UserService:retrieval",
                "entity:UserService",
                kind="retrieval-summary",
                provenance="llm",
                confidence=0.85,
            )
        )

        payload = graph.to_json()
        serialized_payload = json.dumps(payload, ensure_ascii=False)

        self.assertIn('"schema_version": "rigel.graph-ir.v1"', serialized_payload)
        self.assertEqual(
            {node["type"] for node in payload["nodes"]},
            {node_type.value for node_type in NodeType},
        )
        self.assertEqual(
            {edge["type"] for edge in payload["edges"]},
            {edge_type.value for edge_type in EdgeType},
        )
        dependency_edge = next(edge for edge in payload["edges"] if edge["type"] == "DEPENDS_ON")
        self.assertEqual(dependency_edge["properties"]["kind"], "calls")
        self.assertEqual(dependency_edge["properties"]["provenance"], "lsp")
        self.assertEqual(dependency_edge["properties"]["confidence"], 0.98)


if __name__ == "__main__":
    unittest.main()
