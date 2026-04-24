import unittest

from rigel_demo import EdgeType, JavaParseRequest, NodeType, parse_java_file


class JavaParserTest(unittest.TestCase):
    def test_parse_java_file_builds_physical_structure_graph(self) -> None:
        source = """package com.example;

public class UserService {
  private final UserRepository repo;
  private int hits, misses = 1;

  public UserService(UserRepository repo) {
    this.repo = repo;
  }

  public String findName(long id) {
    class ResultMapper {
      String map() { return "ok"; }
    }
    return repo.findName(id);
  }

  interface Nested {
    void run();
  }

  enum Role {
    ADMIN, USER
  }
}
"""

        graph = parse_java_file(
            source,
            "src/main/java/com/example/UserService.java",
            request=JavaParseRequest(repository_name="rigel", module_name="core", module_root_path="src/main/java"),
        )
        payload = graph.to_json()

        nodes_by_type = _nodes_by_type(payload)
        entities = nodes_by_type[NodeType.ENTITY.value]
        entity_names = {node["properties"]["display_name"] for node in entities}
        self.assertEqual(
            entity_names,
            {"UserService", "repo", "hits", "misses", "findName", "ResultMapper", "map", "Nested", "run", "Role"},
        )

        service = _entity_by_qualified_name(entities, "com.example.UserService")
        self.assertEqual(service["properties"]["kind_norm"], "class")
        self.assertEqual(service["properties"]["kind_raw"], "class_declaration")
        self.assertTrue(service["properties"]["semantic_hash"].startswith("sha256:"))

        method = _entity_by_qualified_name(entities, "com.example.UserService#findName(long)")
        self.assertEqual(method["properties"]["kind_norm"], "method")
        local_class = _entity_by_qualified_name(entities, "com.example.UserService#findName(long).ResultMapper")

        nested_interface = _entity_by_qualified_name(entities, "com.example.UserService.Nested")
        nested_method = _entity_by_qualified_name(entities, "com.example.UserService.Nested#run()")

        contains_edges = [edge for edge in payload["edges"] if edge["type"] == EdgeType.CONTAINS.value]
        self.assertTrue(_has_edge(contains_edges, "repo:rigel", "module:rigel:core"))
        self.assertTrue(
            _has_edge(
                contains_edges,
                "module:rigel:core",
                "file:rigel:src/main/java/com/example/UserService.java",
            )
        )
        self.assertTrue(
            _has_edge(
                contains_edges,
                "file:rigel:src/main/java/com/example/UserService.java",
                service["id"],
            )
        )
        self.assertTrue(_has_edge(contains_edges, method["id"], local_class["id"]))
        self.assertTrue(_has_edge(contains_edges, nested_interface["id"], nested_method["id"]))

        anchors = nodes_by_type[NodeType.ANCHOR.value]
        service_anchor_edges = [
            edge
            for edge in payload["edges"]
            if edge["type"] == EdgeType.HAS_ANCHOR.value and edge["source_id"] == service["id"]
        ]
        service_anchor_roles = {
            next(anchor for anchor in anchors if anchor["id"] == edge["target_id"])["properties"]["role"]
            for edge in service_anchor_edges
        }
        self.assertEqual(service_anchor_roles, {"definition", "body"})

        file_node = nodes_by_type[NodeType.FILE.value][0]
        self.assertEqual(file_node["properties"]["language"], "java")
        self.assertTrue(file_node["properties"]["content_hash"].startswith("sha256:"))

    def test_semantic_hash_ignores_comments_and_whitespace(self) -> None:
        compact_source = "package demo; class User { int id; String name(){return \"a\";} }"
        formatted_source = """package demo;

// 这个注释不应影响语义哈希
class User {
  int id;

  String name() {
    return "a";
  }
}
"""

        request = JavaParseRequest(repository_name="rigel")
        compact_graph = parse_java_file(compact_source, "User.java", request=request).to_json()
        formatted_graph = parse_java_file(formatted_source, "User.java", request=request).to_json()
        compact_entities = _nodes_by_type(compact_graph)[NodeType.ENTITY.value]
        formatted_entities = _nodes_by_type(formatted_graph)[NodeType.ENTITY.value]

        compact_user = _entity_by_qualified_name(compact_entities, "demo.User")
        formatted_user = _entity_by_qualified_name(formatted_entities, "demo.User")
        self.assertEqual(compact_user["properties"]["semantic_hash"], formatted_user["properties"]["semantic_hash"])

        compact_file = _nodes_by_type(compact_graph)[NodeType.FILE.value][0]
        formatted_file = _nodes_by_type(formatted_graph)[NodeType.FILE.value][0]
        self.assertNotEqual(compact_file["properties"]["content_hash"], formatted_file["properties"]["content_hash"])


def _nodes_by_type(payload: dict) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for node in payload["nodes"]:
        grouped.setdefault(node["type"], []).append(node)
    return grouped


def _entity_by_qualified_name(entities: list[dict], qualified_name: str) -> dict:
    return next(node for node in entities if node["properties"]["qualified_name"] == qualified_name)


def _has_edge(edges: list[dict], source_id: str, target_id: str) -> bool:
    return any(edge["source_id"] == source_id and edge["target_id"] == target_id for edge in edges)


if __name__ == "__main__":
    unittest.main()
