# Rigel Demo

Rigel Demo 是一个本地代码语义图谱构建与问答演示项目。它面向 Java 仓库，通过解析源码抽取项目代码层级、类、方法及字段等实体，并建立依赖、继承等语义关系图谱。结合检索增强（RAG）技术，为代码问答、架构理解和影响范围分析提供精准、可追溯的上下文基础。

## 核心特性

- **本地化端到端工作流**：提供 `init`、`index`、`web` 等 CLI 命令，轻松管理配置、索引状态与本地图数据库。
- **全方位语义补全**：基于 Tree-sitter 提取代码结构，并利用 Java LSP (multilspy) 深入补全跨文件的调用、实现、继承等关联关系。
- **高效摘要与向量检索**：使用 FalkorDBLite 持久化图谱，为模块、文件和核心实体生成摘要与 Embedding，支持自然语言到图谱节点的高效映射。
- **增量索引支持**：基于文件哈希智能识别代码变动，仅对增删改的文件进行索引更新。
- **可视化交互界面**：提供 Web 界面，左侧动态展示代码关系图谱，右侧提供基于 LangGraph 的智能问答助手。

## 技术栈

- **后端底座**：Python 3.13+, FastAPI, LangGraph, 兼容 OpenAI SDK 接口
- **代码解析**：Tree-sitter / tree-sitter-java, multilspy
- **图数据库**：FalkorDBLite
- **前端界面**：React 19, Vite, Tailwind CSS, react-force-graph-2d

## 快速开始

### 1. 安装项目依赖

在 Rigel Demo 项目目录下安装 Python 及前端依赖：

```bash
uv sync
# 前端依赖会在首次运行 rigel web 时自动检查并安装，也可手动执行：
# cd src/rigel_demo/web/frontend && npm install
```

### 2. 初始化工作区

进入您希望分析的目标 Java 项目目录（支持 Maven/Gradle 或普通源码）：

```bash
cd /path/to/your/java-repository
```

执行初始化，在目标项目内生成配置文件 `.rigel/config.json`：

```bash
# 如果是在源码环境下运行：
uv run --project /path/to/rigel-demo rigel init

# (如果已全局安装本项目，可直接使用 rigel init)
```

### 3. 配置模型参数

编辑目标项目生成的 `.rigel/config.json`。默认配置兼容 OpenAI 接口，预留了 Rerank 节点供使用（如 Gitee AI）：

- `chat`: 问答助手模型配置（如 `gpt-4o`，需填写 `api_key` 等）
- `summary`: 为代码生成摘要的模型配置
- `embedding`: 向量化模型配置（如 `text-embedding-3-small`）
- `rerank`: 重排模型配置（可选，推荐配置 `/rerank` 兼容接口）

### 4. 构建代码图谱

在目标 Java 项目下执行索引构建，解析代码并持久化图谱数据：

```bash
uv run --project /path/to/rigel-demo rigel index
```

*提示：后续如代码有少量修改，可添加 `--incremental` 参数进行增量更新。*

### 5. 启动 Web 演示

```bash
uv run --project /path/to/rigel-demo rigel web
```

启动后将自动在浏览器打开 `http://127.0.0.1:5000`，即可通过可视化界面体验代码图谱与智能问答。
