from __future__ import annotations

from unittest import TestCase

from rigel_demo.graph import EdgeType, GraphEdge


class GraphIRContractTest(TestCase):
    def test_graph_edge_requires_semantic_suffix(self) -> None:
        with self.assertRaisesRegex(ValueError, "kind 或 role"):
            GraphEdge.create(EdgeType.CONTAINS, "source", "target")

    def test_graph_edge_rejects_empty_semantic_suffix(self) -> None:
        with self.assertRaisesRegex(ValueError, "非空字符串"):
            GraphEdge.create(EdgeType.CONTAINS, "source", "target", kind="")
