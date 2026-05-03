from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from rigel_demo.core import EdgeType, NodeType
from rigel_demo.java import JavaParseRequest, JavaSemanticEdgeRequest, enrich_java_semantic_edges, parse_java_file
from rigel_demo.java.requests import GENERATED_ZONE


class JavaSemanticEdgesTest(TestCase):
    def test_lsp_references_use_name_anchor_and_create_reference_edges(self) -> None:
        with TemporaryDirectory() as workspace:
            repository_path = Path(workspace)
            service_path = repository_path / "src" / "main" / "java" / "demo" / "Service.java"
            caller_path = repository_path / "src" / "main" / "java" / "demo" / "Caller.java"
            service_source = "package demo;\npublic class Service {}\n"
            caller_source = "package demo;\npublic class Caller {\n  private Service service;\n}\n"
            service_path.parent.mkdir(parents=True)
            service_path.write_text(service_source, encoding="utf-8")
            caller_path.write_text(caller_source, encoding="utf-8")
            graph = parse_java_file(
                service_source,
                "src/main/java/demo/Service.java",
                request=JavaParseRequest(repository_name="demo"),
            )
            _merge_graph(
                graph,
                parse_java_file(
                    caller_source,
                    "src/main/java/demo/Caller.java",
                    request=JavaParseRequest(repository_name="demo"),
                ),
            )
            fake_lsp = _FakeLspClient(reference_location=_location("src/main/java/demo/Caller.java", 2, 10))

            enrich_java_semantic_edges(
                graph,
                request=JavaSemanticEdgeRequest(repository_root_path=str(repository_path)),
                lsp_client=fake_lsp,
            )

        service_entity = _entity_by_display_name(graph, "Service")
        reference_edges = [
            edge
            for edge in graph.edges
            if edge.type == EdgeType.DEPENDS_ON
            and edge.target_id == service_entity.id
            and edge.properties.get("kind") == "references"
        ]
        service_name_anchor = _anchor_by_role(graph, service_entity.id, "name")
        self.assertIn(
            (
                "src/main/java/demo/Service.java",
                int(service_name_anchor.properties["start_line"]) - 1,
                int(service_name_anchor.properties["start_col"]) - 1,
            ),
            fake_lsp.reference_requests,
        )
        self.assertTrue(reference_edges)
        self.assertEqual(reference_edges[0].properties["provenance"], "lsp")

    def test_generated_duplicate_entities_create_alias_edges(self) -> None:
        with TemporaryDirectory() as workspace:
            repository_path = Path(workspace)
            production_path = repository_path / "src" / "main" / "java" / "demo" / "PaymentService.java"
            generated_path = repository_path / "target" / "generated-sources" / "demo" / "PaymentService.java"
            production_source = "package demo;\npublic class PaymentService {}\n"
            generated_source = "package demo;\npublic class PaymentService {}\n"
            production_path.parent.mkdir(parents=True)
            generated_path.parent.mkdir(parents=True)
            production_path.write_text(production_source, encoding="utf-8")
            generated_path.write_text(generated_source, encoding="utf-8")
            graph = parse_java_file(
                production_source,
                "src/main/java/demo/PaymentService.java",
                request=JavaParseRequest(repository_name="demo"),
            )
            _merge_graph(
                graph,
                parse_java_file(
                    generated_source,
                    "target/generated-sources/demo/PaymentService.java",
                    request=JavaParseRequest(repository_name="demo", file_zone=GENERATED_ZONE),
                ),
            )

            enrich_java_semantic_edges(
                graph,
                request=JavaSemanticEdgeRequest(repository_root_path=str(repository_path)),
                lsp_client=_FakeLspClient(),
            )

        alias_edges = [edge for edge in graph.edges if edge.type == EdgeType.ALIASES]
        self.assertEqual(len(alias_edges), 1)
        self.assertEqual(alias_edges[0].properties["kind"], "generated-mirror")
        self.assertEqual(alias_edges[0].properties["confidence"], 1.0)
        generated_entities = [
            node
            for node in graph.nodes
            if node.type == NodeType.ENTITY
            and node.properties["qualified_name"] == "demo.PaymentService"
            and node.properties["origin"] == "generated"
        ]
        self.assertEqual(alias_edges[0].source_id, generated_entities[0].id)

    def test_lsp_requests_convert_tree_sitter_byte_column_to_utf16_column(self) -> None:
        with TemporaryDirectory() as workspace:
            repository_path = Path(workspace)
            source_path = repository_path / "src" / "main" / "java" / "demo" / "UnicodeService.java"
            source = 'package demo;\npublic class UnicodeService {\n  private String 中文 = "值"; Service service;\n}\nclass Service {}\n'
            source_path.parent.mkdir(parents=True)
            source_path.write_text(source, encoding="utf-8")
            graph = parse_java_file(
                source,
                "src/main/java/demo/UnicodeService.java",
                request=JavaParseRequest(repository_name="demo"),
            )
            fake_lsp = _FakeLspClient()

            enrich_java_semantic_edges(
                graph,
                request=JavaSemanticEdgeRequest(repository_root_path=str(repository_path)),
                lsp_client=fake_lsp,
            )

        service_definition_requests = [
            request
            for request in fake_lsp.definition_requests
            if request[0] == "src/main/java/demo/UnicodeService.java" and request[1] == 2
        ]
        self.assertIn(("src/main/java/demo/UnicodeService.java", 2, 27), service_definition_requests)
        self.assertNotIn(("src/main/java/demo/UnicodeService.java", 2, 33), service_definition_requests)


class _FakeLspClient:
    def __init__(self, *, reference_location: dict[str, object] | None = None) -> None:
        self.reference_location = reference_location
        self.definition_requests: list[tuple[str, int, int]] = []
        self.reference_requests: list[tuple[str, int, int]] = []

    def request_definition(self, file_path: str, line: int, column: int) -> list[dict[str, object]]:
        self.definition_requests.append((file_path, line, column))
        return []

    def request_references(self, file_path: str, line: int, column: int) -> list[dict[str, object]]:
        self.reference_requests.append((file_path, line, column))
        return [self.reference_location] if self.reference_location is not None else []


def _merge_graph(target, source) -> None:
    existing_node_ids = {node.id for node in target.nodes}
    existing_edge_ids = {edge.id for edge in target.edges}
    for node in source.nodes:
        if node.id not in existing_node_ids:
            target.nodes.append(node)
            existing_node_ids.add(node.id)
    for edge in source.edges:
        if edge.id not in existing_edge_ids:
            target.edges.append(edge)
            existing_edge_ids.add(edge.id)


def _entity_by_display_name(graph, display_name: str):
    return next(
        node
        for node in graph.nodes
        if node.type == NodeType.ENTITY and node.properties["display_name"] == display_name
    )


def _anchor_by_role(graph, owner_id: str, role: str):
    anchor_id = next(
        edge.target_id
        for edge in graph.edges
        if edge.type == EdgeType.HAS_ANCHOR and edge.source_id == owner_id and edge.properties["role"] == role
    )
    return next(node for node in graph.nodes if node.id == anchor_id)


def _location(relative_path: str, line: int, column: int) -> dict[str, object]:
    return {
        "uri": f"file:///{relative_path}",
        "absolutePath": f"/repo/{relative_path}",
        "relativePath": relative_path,
        "range": {
            "start": {"line": line, "character": column},
            "end": {"line": line, "character": column + 1},
        },
    }
