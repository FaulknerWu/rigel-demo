"""CLI 配置文档的默认值和 Web 段校验。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rigel_demo.config.document import ConfigDocumentErrorMessages, read_config_document

WEB_CONFIG_SECTION_NAME = "web"


@dataclass(frozen=True, slots=True)
class WebConfig:
    """Web 演示后端运行配置。"""

    host: str
    port: int
    open_browser: bool

    @classmethod
    def from_repository(cls, repository_path: Path) -> "WebConfig":
        """从目标仓库 `.rigel/config.json` 读取 Web 启动配置。"""

        config_data = _read_web_config(repository_path)
        return cls(
            host=_require_web_string(config_data, "host"),
            port=_require_web_port(config_data.get("port")),
            open_browser=_require_web_bool(config_data.get("open_browser"), "open_browser"),
        )


class WebConfigurationError(ValueError):
    """Web 配置不可用。"""


def _read_web_config(repository_path: Path) -> dict[str, Any]:
    """读取 `.rigel/config.json` 中的 Web 配置段。"""

    config_document = read_config_document(
        repository_path,
        messages=ConfigDocumentErrorMessages(
            missing="未找到配置文件，请先执行 rigel init 并填写配置：{config_path}",
            read="读取 Web 配置文件失败：{config_path}",
            invalid_json="Web 配置文件不是合法 JSON：{config_path}",
            root="Web 配置文件根节点必须是 JSON 对象",
        ),
        missing_error_type=FileNotFoundError,
        error_type=WebConfigurationError,
    )
    web_config = config_document.get(WEB_CONFIG_SECTION_NAME)
    if not isinstance(web_config, dict):
        raise WebConfigurationError(f"配置文件必须包含对象字段：{WEB_CONFIG_SECTION_NAME}")
    return web_config


def _require_web_string(config_data: dict[str, Any], name: str) -> str:
    value = config_data.get(name)
    if not isinstance(value, str) or not value.strip():
        raise WebConfigurationError(f"web.{name} 必须是非空字符串")
    return value.strip()


def _require_web_port(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WebConfigurationError("web.port 必须是整数")
    if value < 1 or value > 65_535:
        raise WebConfigurationError("web.port 必须在 1 到 65535 之间")
    return value


def _require_web_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise WebConfigurationError(f"web.{name} 必须是布尔值")
    return value
