from __future__ import annotations

from unittest import TestCase

from rigel_demo.graph import EdgeType, GraphEdge, GraphNode, NodeType


class GraphIRContractTest(TestCase):
    def test_graph_node_requires_non_empty_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "GraphNode.id"):
            GraphNode(id=" ", type=NodeType.ENTITY, properties={})

    def test_graph_edge_requires_semantic_suffix(self) -> None:
        with self.assertRaisesRegex(ValueError, "kind 或 role"):
            GraphEdge.create(EdgeType.CONTAINS, "source", "target")

    def test_graph_edge_rejects_empty_semantic_suffix(self) -> None:
        with self.assertRaisesRegex(ValueError, "非空字符串"):
            GraphEdge.create(EdgeType.CONTAINS, "source", "target", kind="")

    def test_graph_edge_requires_non_empty_endpoints(self) -> None:
        with self.assertRaisesRegex(ValueError, "GraphEdge.source_id"):
            GraphEdge.create(EdgeType.CONTAINS, " ", "target", kind="physical-membership")

        with self.assertRaisesRegex(ValueError, "GraphEdge.target_id"):
            GraphEdge.create(EdgeType.CONTAINS, "source", "", kind="physical-membership")

    def test_graph_edge_rejects_invalid_confidence(self) -> None:
        invalid_confidences = [True, "1.0", -0.1, 1.1]

        for confidence in invalid_confidences:
            with self.subTest(confidence=confidence):
                with self.assertRaisesRegex(ValueError, "GraphEdge.confidence"):
                    GraphEdge(
                        id="edge:test",
                        type=EdgeType.CONTAINS,
                        source_id="source",
                        target_id="target",
                        properties={
                            "kind": "physical-membership",
                            "confidence": confidence,
                        },
                    )
