import tempfile
import unittest
from pathlib import Path

from rigel_demo import EdgeType, GraphIR, JavaParseRequest, JavaSemanticEdgeRequest, enrich_java_semantic_edges, parse_java_file


class FakeJavaLspClient:
    def __init__(self, definitions: dict[tuple[str, int, int], list[dict]] | None = None) -> None:
        self.definitions = definitions or {}

    def request_definition(self, file_path: str, line: int, column: int) -> list[dict]:
        return self.definitions.get((file_path, line, column), [])

    def request_references(self, file_path: str, line: int, column: int) -> list[dict]:
        return []

    def request_hover(self, relative_file_path: str, line: int, column: int) -> dict | None:
        return None


class JavaSemanticEdgesTest(unittest.TestCase):
    def test_enrich_java_semantic_edges_uses_lsp_and_tree_sitter_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            root = Path(workspace)
            service_path = "src/main/java/demo/Service.java"
            repository_path = "src/main/java/demo/Repository.java"
            runner_path = "src/main/java/demo/Runner.java"
            base_path = "src/main/java/demo/Base.java"
            service_source = """package demo;

public class Service extends Base implements Runner {
  private Repository repository;

  @Override
  public String run(User user) {
    return repository.findName(user.id);
  }
}
"""
            repository_source = """package demo;

public class Repository {
  public String findName(long id) {
    return "ok";
  }
}
"""
            runner_source = """package demo;

public interface Runner {
  String run(User user);
}
"""
            base_source = """package demo;

public class Base {
}
"""
            for relative_path, source in {
                service_path: service_source,
                repository_path: repository_source,
                runner_path: runner_source,
                base_path: base_source,
            }.items():
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(source, encoding="utf-8")

            graph = GraphIR()
            request = JavaParseRequest(repository_name="rigel", module_name="core", module_root_path="src/main/java")
            for relative_path, source in {
                service_path: service_source,
                repository_path: repository_source,
                runner_path: runner_source,
                base_path: base_source,
            }.items():
                parsed_graph = parse_java_file(source, relative_path, request=request)
                graph.nodes.extend(parsed_graph.nodes)
                graph.edges.extend(parsed_graph.edges)

            repository_method = _entity_by_qualified_name(graph, "demo.Repository#findName(long)")
            lsp_client = FakeJavaLspClient(
                {
                    (service_path, 7, 22): [
                        _location(repository_path, 4, 17),
                    ]
                }
            )

            enrich_java_semantic_edges(
                graph,
                request=JavaSemanticEdgeRequest(repository_root_path=str(root)),
                lsp_client=lsp_client,
            )

            payload = graph.to_json()
            service_method = _entity_by_qualified_name(graph, "demo.Service#run(User)")
            service_class = _entity_by_qualified_name(graph, "demo.Service")
            base_class = _entity_by_qualified_name(graph, "demo.Base")
            runner_interface = _entity_by_qualified_name(graph, "demo.Runner")
            runner_method = _entity_by_qualified_name(graph, "demo.Runner#run(User)")

            call_edge = _edge(payload, EdgeType.DEPENDS_ON, service_method["id"], repository_method["id"], "calls")
            self.assertEqual(call_edge["properties"]["provenance"], "lsp")
            self.assertGreaterEqual(call_edge["properties"]["confidence"], 0.9)

            extends_edge = _edge(payload, EdgeType.SPECIALIZES, service_class["id"], base_class["id"], "extends")
            self.assertEqual(extends_edge["properties"]["provenance"], "tree-sitter")
            self.assertLess(extends_edge["properties"]["confidence"], 0.8)

            implements_edge = _edge(
                payload,
                EdgeType.SPECIALIZES,
                service_class["id"],
                runner_interface["id"],
                "implements",
            )
            self.assertEqual(implements_edge["properties"]["provenance"], "tree-sitter")

            override_edge = _edge(payload, EdgeType.SPECIALIZES, service_method["id"], runner_method["id"], "override")
            self.assertEqual(override_edge["properties"]["provenance"], "tree-sitter")


def _location(relative_path: str, line: int, character: int) -> dict:
    return {
        "uri": f"file:///{relative_path}",
        "relativePath": relative_path,
        "absolutePath": f"/{relative_path}",
        "range": {"start": {"line": line - 1, "character": character - 1}, "end": {"line": line - 1, "character": character}},
    }


def _entity_by_qualified_name(graph: GraphIR, qualified_name: str) -> dict:
    return next(
        node.to_json()
        for node in graph.nodes
        if node.type.value == "Entity" and node.properties["qualified_name"] == qualified_name
    )


def _edge(payload: dict, edge_type: EdgeType, source_id: str, target_id: str, kind: str) -> dict:
    return next(
        edge
        for edge in payload["edges"]
        if edge["type"] == edge_type.value
        and edge["source_id"] == source_id
        and edge["target_id"] == target_id
        and edge["properties"].get("kind") == kind
    )


if __name__ == "__main__":
    unittest.main()
