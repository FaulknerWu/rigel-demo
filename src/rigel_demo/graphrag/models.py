"""GraphRAG chat 对外模型与异常类型。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from rigel_demo.config import LLMConfig
from rigel_demo.llm import LLMMessage


class RigelGraphRAGError(RuntimeError):
    """GraphRAG 调用失败。"""


@dataclass(frozen=True, slots=True)
class GraphRAGTrace:
    """前端展示用 GraphRAG 查询轨迹。"""

    name: str
    args: dict[str, object]


@dataclass(frozen=True, slots=True)
class GraphRAGReply:
    """GraphRAG 生成结果。"""

    content: str
    traces: list[GraphRAGTrace]


class RigelChatService(Protocol):
    config: LLMConfig

    def send_messages(self, messages: Sequence[LLMMessage]) -> GraphRAGReply:
        """按对话历史生成回答。"""
