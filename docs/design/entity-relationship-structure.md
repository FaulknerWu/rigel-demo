# Rigel 实体与关系结构设计规范

> 说明：本文档描述当前采用的目标 Schema 规范，不代表 `core/` 目录中的 Rust 数据模型已经完全对齐到该结构。

## 1. 核心设计理念

- **代码原文剥离**：图数据库中不存储代码原文。完整的代码保存在 Git 或对象存储中，图数据库只保留身份标识（Identity）、关系（Relations）、哈希值（Hashes）、摘要（Summaries）和位置锚点（Anchors）。用到时再进行懒加载。
- **粗粒度关系优先**：避免将图数据库退化为抽象语法树（AST）或 Token 级别的细粒度图。专注于宏观语义导航（如文件、模块、类、方法的依赖与包含关系）。
- **全量静态解析**：作为目标首版的约束，图谱设计默认以离线全量静态索引为前提，暂不涉及增量更新与快照级别多版本管理（版本管理设计过于复杂，留作后续按独立版本迭代论证）。

## 2. 节点设计

### 2.1 基础设施节点

- **Repository（代码库）**
  - `repo_id`: 唯一标识。
  - `name`: 代码库名称。

### 2.2 物理代码结构节点

- **Module（模块）**：单体仓库（Monorepo）的物理或逻辑边界。
  - `module_id`, `name`, `root_path`
  - `ecosystem`: 生态系统（如 npm, maven, pip）。
  - `zone`: 所属区域划分（例如：`prod` 生产代码, `test` 测试, `tooling` 工具, `vendor` 第三方依赖）。
- **File（文件）**：一等公民节点，承载文档级别的操作。
  - `file_id`, `relative_path` (相对路径), `language` (语言)
  - `zone`: 继承或覆盖模块的区域划分。
  - `content_hash`: 文件的完整内容哈希（用于检测文件级变更）。
  - `position_encoding`: 字符编码格式（如 UTF-8, UTF-16）。*补充解释：不同语言的提取器计算行列号的编码标准不同，必须在此声明以保证后续切片提取的精确性。*

### 2.3 核心语义节点

- **Entity（代码语义实体）**：图的真正语义中心。涵盖类、函数、接口、枚举、全局变量、路由等。
  - `entity_id`
  - `entity_key`: 签名键。
  - `display_name` (展示名), `qualified_name` (全限定名)
  - `kind_norm`: 归一化类型（如 `function`, `class`）。
  - `kind_raw`: 提取器提供的原始类型（保留特定语言的细节）。
  - `origin`: 来源类型（`internal` 内部代码 | `external` 外部依赖 | `generated` 生成代码）。
  - `semantic_hash`: 语义哈希。*补充解释：由实体的签名和主体内容计算得出，忽略空格或注释的变化。用于精准控制摘要和向量的缓存失效。*

### 2.4 辅助元数据节点

- **Anchor（位置锚点）**：记录代码的具体范围（Span）。
  - `anchor_id`, `start_line`, `start_col`, `end_line`, `end_col`
  - `role`: 角色（例如是“定义区域”还是“完整方法体区域”）。
  - *补充解释：不把每一行代码变成图节点，而是用 Anchor 记录范围。Agent 找到对应的实体后，再根据 Anchor 的坐标去对象存储中拉取代码原文。*
- **Summary（摘要 - 向量检索的入口）**：**这是整个系统的第一检索表面。**
  - `summary_id`, `text` (摘要文本内容)
  - `purpose`: 摘要用途（`retrieval` 用于检索 | `rollup` 用于向上汇总）。
  - `source_hash`: 关联的原始实体的语义哈希。
  - `embedding_model`: 生成向量的模型名称。
  - `embedding`: **向量数据**（Vector）。

## 3. 边设计

为了控制图的复杂度和查询性能，将图数据库的关系收敛为 **6 种核心关系族**。语言特有的细节（如调用、读、写）通过属性 `kind` 进行下钻描述，而不在顶层创建几百种关系类型。

1. **`CONTAINS` (包含关系)**
   - **路径**：`Repository -> Module -> File -> Entity -> Nested-Entity`
   - **作用**：表达所有层级结构。
2. **`DEPENDS_ON` (依赖关系)**
   - **路径**：`Entity -> Entity`
   - **属性**：`kind` (如 calls, imports, type-use, schema-usage 等), `provenance` (来源), `confidence` (置信度，处理多语言混合提取时的不确定性)。
   - **作用**：覆盖绝大部分的代码语义使用场景。
3. **`SPECIALIZES` (特化/派生关系)**
   - **路径**：`Entity -> Entity`
   - **属性**：同上。
   - **作用**：处理面向对象中的继承、接口实现、重写（Override）等。
4. **`ALIASES` (别名关系)**
   - **路径**：`Entity -> Entity`
   - **作用**：处理重导出（Re-exports）、导入别名或生成的镜像代码（同一个语义对象的不同包装）。
5. **`HAS_ANCHOR` (关联锚点)**
   - **路径**：`Entity -> Anchor`, `File -> Anchor`
   - **属性**：`role`
   - **作用**：连接语义节点与其在源代码中的确切物理位置。
6. **`DESCRIBES` (描述关系)**
   - **路径**：`Summary -> Entity | File | Module`
   - **作用**：将生成的自然语言摘要（及向量）挂载到对应的实体、文件或模块上。

## 4. 业务场景与检索链路

一次典型的检索分为以下几步：

1. **语义召回 (Recall)**：用户的自然语言 query 转为向量，直接与 `Summary.embedding` 进行距离计算，命中最相关的 `Summary` 节点。
2. **锁定种子 (Seed Targeting)**：通过 `(Summary)-[:DESCRIBES]->(Entity|File)` 关系，迅速找到对应的代码实体或文件（此时可利用 `zone` 属性过滤掉测试代码或 vendor 代码）。
3. **图谱扩充 (Expansion)**：
   - 解析 `ALIASES`（合并别名）。
   - 通过 `CONTAINS` 上下文向上/下跳跃 1 步。
   - 通过 `DEPENDS_ON` 找到核心的调用和被调用者。
   - 通过 `SPECIALIZES` 找到相关的接口和实现类。
4. **组装返回 (Assembly)**：系统返回摘要文本、实体身份、文件路径以及 Anchor 坐标。外部的 Agent 或应用层根据 Anchor 坐标去集中存储中懒加载具体的代码片段进行阅读。
