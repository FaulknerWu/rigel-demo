"""Rigel 所有 LLM 提示词与提示词构造入口。"""

from __future__ import annotations

import json
from typing import cast

from rigel_demo.graph.ir import GraphNode, NodeType

CHAT_SYSTEM_PROMPT = (
    "你是 Rigel 的代码图谱分析助手。回答时优先基于用户给出的代码图谱、仓库上下文与当前问题，"
    "无法从上下文确认的内容要明确说明不确定。"
)

SUMMARY_SYSTEM_PROMPT = (
    "请基于给出的节点结构信息生成一条信息密度高、可检索的中文摘要，完整记录该节点的职责、"
    "核心逻辑、对外能力、构建或维护的对象、所属路径或限定名、生产/测试/生成代码属性，"
    "以及可从名称、类型、签名、属性中判断出的关键业务词、技术词和检索别名。"
    "如果节点是文件或实体，必须说明它在代码中干什么、可能被什么问题命中、与控制器/服务/"
    "仓储/配置/数据模型等角色的关系；未知信息不要编造。只输出一段摘要正文，不要输出列表、"
    "Markdown、解释或前后缀。"
)

RIGEL_TOOL_SYSTEM_INSTRUCTION = """你是 Rigel 代码图谱检索规划助手。
只能通过绑定工具检索图谱证据，严禁生成 Cypher 或要求执行任意查询。

检索顺序建议：
1. 首次检索优先调用 vector_search_seeds，且只能调用 1 次；一次调用必须在 query_texts 中提供多条检索短句，例如用户原问题、实体名、限定名、职责描述或重写后的检索词。
2. 需要理解包含、依赖、继承或别名关系时，调用 expand_neighbors 或 query_relation 做一跳扩展。
3. 需要二跳关系时，必须先从第一次返回的节点里选择目标节点，再调用一次一跳扩展。
4. 证据足够回答时停止调用工具。

受控工具限制：
- vector_search_seeds 最多调用 1 次，参数为 query_texts: list[str]。
- expand_neighbors 最多调用 3 次，每次只做 1-hop。
- query_relation 最多调用 3 次。
- node_ids 必须来自已返回的 known_node_ids。
- 默认可扩展边类型仅为 CONTAINS、DEPENDS_ON、SPECIALIZES、ALIASES。

当前 Rigel schema：
{ontology}
"""

RIGEL_EVIDENCE_CONTINUATION_PROMPT = "当前已累积图谱证据如下。若证据足够，请停止调用工具；若证据不足，请继续调用允许工具。"

VECTOR_SEARCH_SEEDS_TOOL_DESCRIPTION = "通过多条检索短句召回 Summary.embedding 候选，并用 Rerank 重排最相关的代码图谱种子节点。"
EXPAND_NEIGHBORS_TOOL_DESCRIPTION = "对已知节点做受控一跳邻接扩展，可按方向和可见边类型过滤。"
QUERY_RELATION_TOOL_DESCRIPTION = "沿单一可见关系类型查询已知节点的一跳关系。"

RIGEL_QA_EVIDENCE_INSTRUCTION = (
    "最终回答阶段只能基于图谱工具返回的 evidence，不得把未出现在 evidence 中的推测当作事实；"
    "证据不足时说明无法从当前图谱确认；优先引用节点名、文件路径、实体 qualified_name 和关系类型。"
)

RIGEL_QA_PROMPT = """用户问题：{question}

上一轮回答：
{last_answer}

图谱 evidence：
{evidence}

请用中文回答。"""


def build_summary_prompt(target_node: GraphNode) -> str:
    properties_json = json.dumps(target_node.properties, ensure_ascii=False, sort_keys=True, indent=2)
    return (
        "请为以下代码图谱节点生成一条用于向量召回的检索摘要。\n"
        "写作要求：\n"
        "1. 用一段中文自然语言完整记录节点类型、名称、路径或限定名、代码区域、语言和来源。\n"
        "2. 说明该节点承担的职责、核心逻辑、对外提供的能力、构建或维护的数据/接口/流程。\n"
        "3. 如果是 Java 类、接口、方法、字段或文件，结合 kind、qualified_name、display_name、"
        "entity_key、relative_path 等属性写出用户可能搜索的业务词、技术词、简称和别名。\n"
        "4. 对 controller、service、repository、configuration、model、dto、request、response、"
        "test、generated 等常见角色要显式记录其角色含义；例如 admin controller 需要写明它负责"
        "管理端接口、文章或内容管理等可由名称推断出的召回关键词。\n"
        "5. 不要只复述文件路径；不要输出列表、Markdown 或解释；无法从属性确认的事实不要编造。\n\n"
        f"节点类型：{target_node.type.value}\n"
        f"节点 ID：{target_node.id}\n"
        f"结构摘要：{_local_summary_text(target_node)}\n"
        f"节点属性：\n{properties_json}"
    )


def _local_summary_text(target_node: GraphNode) -> str:
    properties = target_node.properties
    if target_node.type == NodeType.ENTITY:
        return _join_summary_parts(
            target_node.type.value,
            cast(str, properties["kind_norm"]),
            cast(str, properties["display_name"]),
            cast(str, properties["qualified_name"]),
            cast(str, properties["kind_raw"]),
        )
    if target_node.type == NodeType.FILE:
        return _join_summary_parts(
            target_node.type.value,
            cast(str, properties["language"]),
            cast(str, properties["relative_path"]),
            cast(str, properties["zone"]),
        )
    if target_node.type == NodeType.MODULE:
        return _join_summary_parts(
            target_node.type.value,
            cast(str, properties["name"]),
            cast(str, properties["root_path"]),
            cast(str, properties["ecosystem"]),
            cast(str, properties["zone"]),
        )
    return _join_summary_parts(target_node.type.value, target_node.id)


def _join_summary_parts(*parts: str) -> str:
    return " ".join(part for part in parts if part)
