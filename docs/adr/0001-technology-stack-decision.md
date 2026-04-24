# ADR-0001: 演示仓库技术栈选型

> 说明：本文档仅记录当前 `rigel-demo` 演示仓库的技术决策。有关 Rigel 长期主线（Rust / FalkorDB 等基础设施）的正式设计，请参阅 Rigel 主仓库。

## 状态
已接受 (Accepted)

## 背景
由于近期展示作品的需要，我们需要一个易于快速开发的原型。考虑到 Java 语言在企业级仓库中的代表性以及其复杂的面向对象特性（如继承、多态、泛型），我们选择以解析 Java 仓库作为演示目标。

## 决策
1. **采用 Python 为核心承载语言**：在 demo 与原型验证阶段，Python 提供无与伦比的快速实验能力及微软开源的 `multilspy` 库，帮助快速构建 LSP，无须手动实现复杂的 JSON-RPC 通信。
2. **集成 Tree-sitter 作为结构骨架**：利用 `tree-sitter` 的 Java 解析能力，以纯静态的方式快速对源码进行拆解、词法分析，并圈定代码块的位置锚点（Anchor），提取文件中具体的实体类型（Class, Method 等）。
3. **集成 Multilspy 作为深度语义引擎**：在纯静态 AST 解析不足以进行跨文件类型推导的场景下，引入微软开源的 `multilspy` 库。基于 Language Server Protocol (LSP)，在后台拉起真正懂 Java 编译规则的语言服务（如 Eclipse JDT LS），从中提取跨文件定义跳转、引用查找和方法调用链。
4. **采用 FalkorDBLite 作为本地图数据库运行时**：在 demo 阶段引入 `FalkorDBLite`，以嵌入式 Python 包的方式启动本地 Redis + FalkorDB 运行时，用于存储和验证代码知识图谱。它避免了额外部署外部数据库服务的成本，同时保留 Cypher 查询能力与后续迁移到远程 FalkorDB 的路径。

## 决策依据

### 为什么联合使用 Tree-sitter 与 Multilspy
* **Tree-sitter 的局限性**：尽管 Tree-sitter 的 AST 可以精确切分当前文件并高亮语法，但它并不包含项目的编译上下文。遇到诸如 `service.execute()` 的多态接口调用时，纯静态方案无法判定 `service` 具体指向哪个实现类。
* **Multilspy 的便利性**：`multilspy` 以 Python 接口极大地降低了我们去和底层 LSP JSON-RPC 通信的复杂度，使我们能将绝大部分精力投入在图谱组装算法上。
* **互补性**：LSP 更加面向交互式查询（查定义、查引用），需要消耗较高内存；Tree-sitter 则更轻量和泛化。二者结合能达到性能与精准度的最优解。

### 为什么演示目标为 Java 仓库
* 强类型、面向对象，抽象层次高，是非常标准且难以仅靠词法拆解搞定的评估语言，具有极好的代表性。Java LSP (JDT LS) 发育得最为成熟稳定。

### 为什么使用 FalkorDBLite
* **降低演示部署成本**：FalkorDBLite 由应用进程管理本地 Redis + FalkorDB 运行时，适合本地开发、原型验证、离线演示与 CI 场景，不需要用户预先安装或维护独立数据库服务。
* **贴合图谱验证目标**：代码知识图谱天然需要节点、边与路径查询能力。FalkorDBLite 保留 FalkorDB 的图查询能力，可直接承载 Rigel Schema 的节点与关系验证。
* **保留主线迁移路径**：FalkorDBLite 与远程 FalkorDB 的能力模型一致，demo 阶段可以先使用轻量本地运行时，后续需要生产化或多人共享时再切换到 FalkorDB Cloud 或自托管 FalkorDB。

## 影响

### 正向影响
* **加速验证**：极短时间内可跑出一个具备高度智能与绝对调用链准确性的代码解析 Pipeline。
* **零幻觉关联**：基于 LSP 提取的调用关系不会有单纯正则或模型猜测导致的上下文幻觉问题。
* **验证 Schema**：用高度精确的 Java LSP 数据去验证之前定义的 Rigel 6 大边族（`DEPENDS_ON`, `SPECIALIZES` 等）抽象设计的合理性。
* **降低运行门槛**：FalkorDBLite 让演示环境可以随 Python 应用启动图数据库，减少外部服务编排与端口配置成本。

### 负向影响
* **资源消耗（冷启动）**：利用 LSP 进行项目的全面 Build 及索引构建相比纯 Tree-sitter 文本解析会带来显著的 CPU 和内存占用。
* **本地运行时边界**：FalkorDBLite 更适合本地原型和演示，不应直接承担生产级、多用户或高并发图数据库职责；进入生产化阶段可能需要切换到 FalkorDB Cloud 或自托管 FalkorDB。
* **主干脱节风险**：未来如果 Rust 主线（以更轻量的静态全局分析器研发为主）迟迟无法达到此项目的准确率，可能需要在主线重新审视决策。
