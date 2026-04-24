# Rigel Demo
本项目 (`rigel-demo`) 是为了快速开发与展示从 Rigel 主项目分离出来的 demo 仓库，后期预计废弃。

## 演示技术栈与依赖

*   **核心环境**: [Python](https://www.python.org/)
*   **语法树解析**: [Tree-sitter](https://tree-sitter.github.io/tree-sitter/) (针对细粒度 AST 和实体剥离)
*   **LSP 封装**: [multilspy](https://github.com/microsoft/multilspy) (针对跨文件分析)
*   **图数据库**: [FalkorDBLite](https://docs.falkordb.com/operations/falkordblite.html) (用于本地保存 GraphIR 节点与语义边)

## FalkorDBLite 写入示例

本项目通过 `FalkorDBStore` 将 `GraphIR` 幂等写入本地 FalkorDBLite。节点会同时带有通用标签
`RigelNode` 和具体类型标签，例如 `Entity`；边类型直接使用 GraphIR 的顶级边类型，
例如 `DEPENDS_ON`。

写入解析结果：

```python
from pathlib import Path

from rigel_demo import FalkorDBConfig, FalkorDBStore, JavaParseRequest, parse_java_file

source_path = Path("src/main/java/demo/Service.java")
graph = parse_java_file(
    source_path.read_text(encoding="utf-8"),
    str(source_path),
    request=JavaParseRequest(repository_name="rigel", module_name="core"),
)

store = FalkorDBStore.connect(FalkorDBConfig(graph_name="rigel", database_path=".rigel/falkordb.db"))
store.upsert_graph(graph)
```

常用查询：

```cypher
MATCH (entity:RigelNode:Entity)-[:DEPENDS_ON]->(target:RigelNode:Entity)
RETURN entity.qualified_name, target.qualified_name
```

Rigel 主仓库正式的架构决策与主线规划见：
- GitHub: <https://github.com/FaulknerWu/Rigel>
