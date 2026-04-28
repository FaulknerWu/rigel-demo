# Rigel Demo
本项目 (`rigel-demo`) 是为了快速开发与展示从 Rigel 主项目分离出来的 demo 仓库，后期预计废弃。

## 演示技术栈与依赖

*   **核心环境**: [Python](https://www.python.org/)
*   **语法树解析**: [Tree-sitter](https://tree-sitter.github.io/tree-sitter/) (针对细粒度 AST 和实体剥离)
*   **LSP 封装**: [multilspy](https://github.com/microsoft/multilspy) (针对跨文件分析)
*   **图数据库**: [FalkorDBLite](https://docs.falkordb.com/operations/falkordblite.html) (用于本地保存 GraphIR 节点与语义边)

## CLI 初始化与索引流程

先在目标仓库生成 `.rigel/config.json`：

```bash
uv run rigel init
```

填好 Chat、Summary 与 Embedding 配置后，再执行索引：

```bash
uv run rigel index
```

`rigel index` 会扫描当前仓库中的 Java 源码，先用 Tree-sitter 构建基础结构图，再通过
`multilspy` 启动真实 Java LSP 补全跨文件定义跳转、引用与调用关系，并重建
`.rigel/falkordb.db`。运行前需确保本机 `java` 命令可用。

## 功能级模型配置

模型配置写在目标仓库 `.rigel/config.json` 中，`rigel init` 会生成默认模板。
当前按功能拆分为三个顶层字段：

- `chat`: Web 聊天面板 `/api/chat` 使用的生成模型。
- `summary`: `rigel index` 生成 `Summary.text` 使用的生成模型。
- `embedding`: `rigel index` 写入 Summary 向量、Web 召回查询向量使用的嵌入模型。

`chat` 与 `summary` 都复用同一套 LLM 客户端字段。`provider` 是提供商名称，
`format` 决定请求协议格式，当前支持 `openai_chat`、`google_generate_content`
与 `openai_responses`。两个功能可以配置不同的模型、密钥、Base URL 与生成参数。

OpenAI Responses API：

```json
{
  "chat": {
    "provider": "openai",
    "format": "openai_responses",
    "model": "gpt-5.2",
    "api_key": "sk-your-openai-key"
  }
}
```

OpenAI Chat Completions API：

```json
{
  "summary": {
    "provider": "openai",
    "format": "openai_chat",
    "model": "gpt-5.2-mini",
    "api_key": "sk-your-openai-key",
    "base_url": "https://api.openai.com/v1",
    "temperature": 0,
    "max_output_tokens": 300
  }
}
```

Google Gemini 原生 generateContent：

```json
{
  "chat": {
    "provider": "google",
    "format": "google_generate_content",
    "model": "gemini-3-flash-preview",
    "api_key": "your-gemini-key"
  }
}
```

新增提供商时设置新的 `chat.provider` 或 `summary.provider` 名称，选择其兼容的
`format`，并通过对应功能段的 `api_key` 与 `base_url` 配置访问参数。可选生成参数包括
`timeout_seconds`、`temperature`、`max_output_tokens` 与 `system_prompt`。

## Embedding 配置

`rigel index` 会调用真实 Embedding API 生成 Summary 向量。配置同样写在目标仓库
`.rigel/config.json` 中的 `embedding` 字段，当前支持 OpenAI-compatible
`/embeddings` 请求格式。

```json
{
  "embedding": {
    "provider": "openai",
    "format": "openai_embeddings",
    "model": "text-embedding-3-small",
    "api_key": "sk-your-openai-key",
    "base_url": null,
    "dimensions": 512,
    "timeout_seconds": 60,
    "batch_size": 64
  }
}
```

`embedding.model`、`embedding.dimensions` 与索引写入的 Summary 向量维度必须和
Web 召回时保持一致；修改后需要重新执行 `rigel index` 重建 `.rigel/falkordb.db`。

## FalkorDBLite 写入示例

本项目通过 `FalkorDBStore` 将 `GraphIR` 幂等写入本地 FalkorDBLite。节点会同时带有通用标签
`RigelNode` 和具体类型标签，例如 `Entity`；边类型直接使用 GraphIR 的顶级边类型，
例如 `DEPENDS_ON`。

写入解析结果：

```python
from pathlib import Path

from rigel_demo.java import JavaParseRequest, parse_java_file
from rigel_demo.storage import FalkorDBConfig, FalkorDBStore

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

## 本地向量召回链路

`rigel index` 会为 Module、File 与 Entity 生成 `Summary` 节点。流程是先调用
`.rigel/config.json` 中 `summary` 配置的生成模型写入 `Summary.text`，再调用
`embedding` 配置的真实 Embedding 模型写入 `Summary.embedding`，并通过 `DESCRIBES`
边连接到被描述的图谱节点。`Summary.summary_model` 和 `Summary.embedding_model`
会分别记录两类模型名称。

Web 后端提供 `/api/recall?q=PaymentService`，流程为：

1. 将用户问题映射到同一套本地向量空间。
2. 读取 `Summary` 节点并按余弦相似度排序，得到召回种子。
3. 通过 `DESCRIBES` 锁定目标 Module、File 或 Entity。
4. 沿 `CONTAINS`、`DEPENDS_ON`、`SPECIALIZES`、`ALIASES` 补充一跳上下文。

`/api/chat` 会优先使用这条向量召回链路组装代码图谱上下文；召回为空时才回退到字段关键词搜索。

Rigel 主仓库正式的架构决策与主线规划见：
- GitHub: <https://github.com/FaulknerWu/Rigel>
