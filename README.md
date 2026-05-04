# Rigel Demo

Rigel Demo 是一个本地代码语义图谱演示项目。它面向 Java 仓库，把源码中的仓库、模块、文件、类、接口、方法、字段等对象抽取为图谱节点，并用包含、依赖、继承、别名、锚点和摘要等关系组织起来，为代码问答、架构理解、调用链追踪和影响范围分析提供可追溯的上下文基础。

项目的核心目标不是把大量源码片段直接塞进大模型上下文，而是先构建可查询的语义图谱，再通过摘要向量召回定位相关节点，沿图谱扩展局部关系，最后按源码锚点懒加载必要代码片段。这样可以减少无关上下文，提高回答依据的可解释性。

## 核心能力

- 本地 CLI 工作流：在目标仓库内执行 `rigel init`、`rigel index`、`rigel web`，统一管理配置、索引状态和本地图数据库。
- Java 代码图谱构建：基于 Tree-sitter Java 提取文件、类型、方法、字段和源码锚点。
- 跨文件语义关系补全：基于 multilspy 启动 Java LSP，补全调用、导入、类型使用、继承、实现、重写等关系。
- 图数据库持久化：使用 FalkorDBLite 在当前仓库 `.rigel/falkordb.db` 保存图谱，并支持 Cypher 查询和向量索引。
- 摘要与向量召回：为 Module、File、Entity 生成检索摘要和 Embedding，通过 Summary 向量索引把自然语言问题映射到图谱节点。
- 演示级增量索引：通过 Java 文件内容哈希识别新增、修改、删除和跳过文件，更新本地图谱。
- Web 演示界面：左侧展示代码图谱，右侧提供基于 GraphRAG-SDK 的自然语言问答入口。

## 技术栈

- Python 3.13+
- FastAPI / Uvicorn
- Tree-sitter / tree-sitter-java
- multilspy
- FalkorDBLite / FalkorDB
- GraphRAG-SDK
- OpenAI SDK 兼容的 Chat Completions 与 Embeddings 接口
- React 19 / Vite / Tailwind CSS / react-force-graph-2d
- uv

## 工作原理

Rigel Demo 的主链路分为五步：

1. 结构解析：扫描目标仓库中的 Java 文件，识别 Maven、Gradle 模块边界和文件分区，生成 Repository、Module、File、Entity、Anchor 等节点。
2. 语义补全：Tree-sitter 先定位可能存在语义关系的位置，LSP 再确认跨文件定义、引用和继承目标，写入 `DEPENDS_ON`、`SPECIALIZES`、`ALIASES` 等边。
3. 图谱持久化：GraphIR 被幂等写入 FalkorDBLite，节点带有 `RigelNode` 通用标签和具体类型标签，关系使用稳定 ID 合并。
4. 摘要索引：系统为模块、文件和实体生成中文检索摘要，写入 Summary 节点，并在 `Summary.embedding` 上创建 FalkorDB 原生向量索引。
5. 检索问答：用户问题先转为向量召回 Summary，再沿 `DESCRIBES` 找到种子节点，并通过图谱邻接关系补充上下文；Web Chat 通过 GraphRAG-SDK 查询同一份本地图谱。

## 数据模型

当前图谱以六类核心节点组织代码仓库：

- `Repository`：代码库根节点。
- `Module`：Maven、Gradle 或默认模块。
- `File`：源码文件，包含相对路径、语言、分区、内容哈希等属性。
- `Entity`：类、接口、枚举、记录、方法、构造函数、字段等语义实体。
- `Anchor`：实体或文件在源码中的位置范围。
- `Summary`：面向检索的自然语言摘要及向量。

顶层关系收敛为六类：

- `CONTAINS`：仓库、模块、文件和实体之间的层级包含。
- `DEPENDS_ON`：调用、导入、引用、类型使用等依赖关系。
- `SPECIALIZES`：继承、接口实现、方法重写等特化关系。
- `ALIASES`：生成代码镜像或重复语义对象之间的别名关系。
- `HAS_ANCHOR`：节点到源码坐标的定位关系。
- `DESCRIBES`：摘要节点到被描述节点的关系。

## 快速开始

### 1. 安装依赖

在本仓库中安装 Python 依赖：

```bash
uv sync
```

Web 前端依赖会在执行 `rigel web` 时自动检查并安装；也可以手动安装：

```bash
cd src/rigel_demo/web/frontend
npm install
```

### 2. 准备目标 Java 仓库

进入你希望分析的 Java 仓库根目录。该仓库可以是 Maven、Gradle 或普通 Java 源码目录：

```bash
cd /path/to/java-repository
```

### 3. 初始化 Rigel 工作目录

```bash
uv run --project /home/user/workspace/rigel-demo rigel init
```

命令会在目标仓库创建 `.rigel/config.json`。该配置不会在已存在时被覆盖。

### 4. 填写模型配置

编辑目标仓库中的 `.rigel/config.json`，至少需要填写 `chat.api_key`、`summary.api_key` 和 `embedding.api_key`。默认配置使用 OpenAI 兼容接口：

```json
{
  "chat": {
    "provider": "openai",
    "model": "gpt-5.2",
    "api_key": "sk-your-openai-key",
    "base_url": null
  },
  "summary": {
    "provider": "openai",
    "model": "gpt-5.2",
    "api_key": "sk-your-openai-key",
    "base_url": null,
    "concurrent_requests": 4
  },
  "embedding": {
    "provider": "openai",
    "format": "openai_embeddings",
    "model": "text-embedding-3-small",
    "api_key": "sk-your-openai-key",
    "base_url": null,
    "dimensions": 512,
    "batch_size": 64,
    "input_mode": "array"
  }
}
```

如果使用兼容 OpenAI 协议的第三方服务，需要同时配置对应段落的 `base_url`。
如果 Embedding 服务只接受单条字符串输入，将 `embedding.input_mode` 改成 `"string"`；标准 OpenAI 兼容批量接口使用 `"array"`。

### 5. 构建索引

```bash
uv run --project /home/wu/workspace/rigel-demo rigel index
```

索引完成后，目标仓库会生成：

- `.rigel/falkordb.db`：FalkorDBLite 本地图数据库。
- `.rigel/falkordb.db.settings`：FalkorDBLite 运行时设置文件。
- `.rigel/rigel.json`：最近一次索引状态和图谱统计。

小规模变更后可以执行演示级增量索引：

```bash
uv run --project /home/wu/workspace/rigel-demo rigel index --incremental
```

### 6. 启动 Web 演示

```bash
uv run --project /home/wu/workspace/rigel-demo rigel web
```

默认访问地址为：

```text
http://127.0.0.1:5000
```

Web 页面左侧展示代码图谱，右侧提供图谱问答。服务启动参数由目标仓库 `.rigel/config.json` 中的 `web.host`、`web.port`、`web.open_browser` 控制。

## 常用命令

```bash
# 初始化当前目标仓库
uv run --project /home/wu/workspace/rigel-demo rigel init

# 全量重建当前目标仓库图谱
uv run --project /home/wu/workspace/rigel-demo rigel index

# 基于已有图数据库执行增量索引
uv run --project /home/wu/workspace/rigel-demo rigel index --incremental

# 构建前端并启动 Web 演示后端
uv run --project /home/wu/workspace/rigel-demo rigel web
```

如果已经在开发环境中安装了本项目，也可以直接使用 `rigel init`、`rigel index` 和 `rigel web`。

## 配置说明

`.rigel/config.json` 按功能分为五段：

- `web`：Web 后端监听地址、端口和是否自动打开浏览器。
- `graphrag`：GraphRAG-SDK 连接 FalkorDB 服务所需的 host、port、username、password。
- `chat`：Web Chat 使用的 Chat Completions 模型配置。
- `summary`：索引阶段生成 Summary 文本的模型配置。
- `embedding`：摘要向量和查询向量使用的 Embedding 模型配置。

`chat` 与 `summary` 会读取各自的 `system_prompt`、`temperature`、`max_output_tokens` 等生成参数。`summary.concurrent_requests` 控制索引阶段并发生成 Summary 文本的请求数。`embedding.format` 当前支持 `openai_embeddings`，`embedding.input_mode` 支持 `array` 和 `string`。
