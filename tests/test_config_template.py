from __future__ import annotations

import json
from pathlib import Path

import pytest

from rigel_demo.cli import init_repository
from rigel_demo.config import LLMConfig, LLMConfigurationError, default_config_document_text
from rigel_demo.config.document import RIGEL_CONFIG_FILE_NAME, RIGEL_WORKSPACE_DIRECTORY_NAME

REQUIRED_CONFIG_SECTIONS = {
    "web",
    "graphrag",
    "chat",
    "summary",
    "embedding",
    "rerank",
}


def test_default_config_template_is_valid_json() -> None:
    config_document = json.loads(default_config_document_text())

    assert isinstance(config_document, dict)
    assert REQUIRED_CONFIG_SECTIONS.issubset(config_document)
    assert "system_prompt" not in config_document["chat"]
    assert "system_prompt" not in config_document["summary"]


def test_init_repository_writes_default_config_template(tmp_path: Path) -> None:
    result = init_repository(tmp_path)
    config_path = tmp_path / RIGEL_WORKSPACE_DIRECTORY_NAME / RIGEL_CONFIG_FILE_NAME

    assert result.config_created is True
    assert config_path.read_text(encoding="utf-8") == default_config_document_text()


def test_llm_config_rejects_external_system_prompt(tmp_path: Path) -> None:
    config_document = json.loads(default_config_document_text())
    config_document["chat"]["system_prompt"] = "custom"
    write_config(tmp_path, config_document)

    with pytest.raises(LLMConfigurationError, match="chat.system_prompt"):
        LLMConfig.from_repository(tmp_path)


def write_config(repository_path: Path, config: dict[str, object]) -> None:
    config_path = repository_path / RIGEL_WORKSPACE_DIRECTORY_NAME / RIGEL_CONFIG_FILE_NAME
    config_path.parent.mkdir()
    config_path.write_text(json.dumps(config), encoding="utf-8")
