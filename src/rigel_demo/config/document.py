"""Rigel 工作区配置文档读取工具。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from json import JSONDecodeError
from pathlib import Path
from typing import Any, NoReturn

RIGEL_WORKSPACE_DIRECTORY_NAME = ".rigel"
RIGEL_CONFIG_FILE_NAME = "config.json"
DEFAULT_CONFIG_RESOURCE_NAME = "default_config.json"


@dataclass(frozen=True, slots=True)
class ConfigDocumentErrorMessages:
    """配置文档读取阶段的错误消息模板。"""

    missing: str
    read: str
    invalid_json: str
    root: str


def config_document_path(repository_path: Path) -> Path:
    """返回目标仓库中的 `.rigel/config.json` 路径。"""

    return repository_path / RIGEL_WORKSPACE_DIRECTORY_NAME / RIGEL_CONFIG_FILE_NAME


def default_config_document_text() -> str:
    """读取随包分发的默认配置模板文本。"""

    return files("rigel_demo.config").joinpath(DEFAULT_CONFIG_RESOURCE_NAME).read_text(encoding="utf-8")


def read_config_document(
    repository_path: Path,
    *,
    messages: ConfigDocumentErrorMessages,
    missing_error_type: type[Exception] | None = None,
    error_type: type[Exception],
) -> dict[str, Any]:
    """读取 `.rigel/config.json`，并按调用方错误类型包装失败信息。"""

    config_path = config_document_path(repository_path)
    if not config_path.exists():
        raise (missing_error_type or error_type)(messages.missing.format(config_path=config_path))

    try:
        config_document = json.loads(config_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise error_type(messages.read.format(config_path=config_path)) from error
    except JSONDecodeError as error:
        raise error_type(messages.invalid_json.format(config_path=config_path)) from error

    if not isinstance(config_document, dict):
        raise error_type(messages.root)
    return config_document


@dataclass(frozen=True, slots=True)
class ConfigFieldReader:
    """按配置段名称生成一致的字段读取与校验错误。"""

    data: dict[str, Any]
    section: str
    error_type: type[Exception]

    def required_string(self, name: str) -> str:
        value = self._require_field(name)
        if isinstance(value, str):
            stripped_value = value.strip()
            if stripped_value:
                return stripped_value
            self._raise(f"缺少必要配置：{self.field_path(name)}")
        self._raise(f"{self.field_path(name)} 必须是字符串")

    def nullable_string(self, name: str) -> str | None:
        value = self._require_field(name)
        if value is None:
            return None
        if not isinstance(value, str):
            self._raise(f"{self.field_path(name)} 必须是字符串或 null")
        stripped_value = value.strip()
        return stripped_value or None

    def positive_float(self, name: str) -> float:
        value = self._require_field(name)
        if not _is_json_number(value):
            self._raise(f"{self.field_path(name)} 必须是数字")
        parsed_value = float(value)
        if parsed_value <= 0:
            self._raise(f"{self.field_path(name)} 必须大于 0")
        return parsed_value

    def nullable_float(self, name: str) -> float | None:
        value = self._require_field(name)
        if value is None:
            return None
        if not _is_json_number(value):
            self._raise(f"{self.field_path(name)} 必须是数字")
        return float(value)

    def positive_int(self, name: str) -> int:
        return self._positive_int_value(self._require_field(name), name)

    def nullable_positive_int(self, name: str) -> int | None:
        value = self._require_field(name)
        if value is None:
            return None
        return self._positive_int_value(value, name)

    def bounded_int(self, name: str, *, minimum: int, maximum: int) -> int:
        value = self._require_field(name)
        if isinstance(value, bool) or not isinstance(value, int):
            self._raise(f"{self.field_path(name)} 必须是整数")
        if value < minimum or value > maximum:
            self._raise(f"{self.field_path(name)} 必须在 {minimum} 到 {maximum} 之间")
        return value

    def field_path(self, name: str) -> str:
        return f"{self.section}.{name}"

    def _require_field(self, name: str) -> object:
        if name not in self.data:
            self._raise(f"缺少必要配置：{self.field_path(name)}")
        return self.data[name]

    def _positive_int_value(self, value: object, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            self._raise(f"{self.field_path(name)} 必须是整数")
        if value <= 0:
            self._raise(f"{self.field_path(name)} 必须大于 0")
        return value

    def _raise(self, message: str) -> NoReturn:
        raise self.error_type(message)


def _is_json_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float)
