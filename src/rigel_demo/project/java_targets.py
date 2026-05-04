"""仓库内 Java 文件索引目标发现。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rigel_demo.java.requests import (
    DEFAULT_MODULE_ECOSYSTEM,
    DEFAULT_MODULE_NAME,
    DEFAULT_ZONE,
    GENERATED_ZONE,
)

IGNORED_DIRECTORY_NAMES = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".rigel",
    ".ruff_cache",
    ".svn",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "out",
    "target",
    "venv",
}

JAVA_SOURCE_ROOT_PATTERNS: tuple[tuple[str, ...], ...] = (
    ("src", "main", "java"),
    ("src", "test", "java"),
    ("src", "generated", "java"),
    ("generated", "src", "main", "java"),
    ("generated-sources",),
)
MAVEN_MARKER_FILE_NAME = "pom.xml"
GRADLE_MARKER_FILE_NAMES = {"build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts"}
TEST_PATH_PARTS = {"test", "tests", "it", "integrationtest", "integration-test"}
TOOLING_PATH_PARTS = {"tool", "tools", "tooling", "script", "scripts", "buildsrc", "build-logic"}
VENDOR_PATH_PARTS = {"vendor", "third_party", "third-party", "external"}
GENERATED_PATH_PARTS = {"generated", "generated-sources", "build", "out", "target"}
GENERATED_SOURCE_MARKER_PARTS = {"generated", "generated-sources"}
HASH_PREFIX = "sha256:"


@dataclass(frozen=True, slots=True)
class JavaFileIndexTarget:
    """单个 Java 文件在仓库内的模块和分区归属。"""

    source_path: Path
    relative_path: str
    module_name: str
    module_root_path: str
    module_ecosystem: str
    module_zone: str
    file_zone: str


def iter_java_targets(repository_path: Path) -> list[JavaFileIndexTarget]:
    # 排序让索引输出在不同文件系统遍历顺序下保持稳定，便于测试和演示复现。
    return sorted(
        (
            _java_file_target(repository_path, path)
            for path in repository_path.rglob("*.java")
            if path.is_file() and not _is_ignored_path(path.relative_to(repository_path))
        ),
        key=lambda target: target.relative_path,
    )


def content_hash(content: bytes) -> str:
    return f"{HASH_PREFIX}{hashlib.sha256(content).hexdigest()}"


def _is_ignored_path(relative_path: Path) -> bool:
    parts = {part.lower() for part in relative_path.parts}
    ignored_parts = parts & IGNORED_DIRECTORY_NAMES
    if not ignored_parts:
        return False
    # target/build/out 通常跳过，但 generated-sources 是 Java 工程里真实可索引的生成源码入口。
    if ignored_parts <= {"build", "out", "target"} and parts & GENERATED_SOURCE_MARKER_PARTS:
        return False
    return True


def _java_file_target(repository_path: Path, source_path: Path) -> JavaFileIndexTarget:
    relative_path = source_path.relative_to(repository_path)
    module_root = _detect_module_root(repository_path, relative_path)
    source_root = _detect_java_source_root(relative_path.relative_to(module_root))
    module_relative_path = relative_path.relative_to(module_root)
    file_zone = _detect_file_zone(module_relative_path, source_root=source_root)
    module_name = _module_name(module_root)
    module_zone = _module_zone(file_zone)
    return JavaFileIndexTarget(
        source_path=source_path,
        relative_path=relative_path.as_posix(),
        module_name=module_name,
        module_root_path=module_root.as_posix(),
        module_ecosystem=_detect_module_ecosystem(repository_path, module_root),
        module_zone=module_zone,
        file_zone=file_zone,
    )


def _detect_module_root(repository_path: Path, relative_path: Path) -> Path:
    parent_parts = relative_path.parts[:-1]
    # 从文件向仓库根回溯，优先选择最近的 Maven/Gradle 标记作为模块根。
    for part_count in range(len(parent_parts), -1, -1):
        candidate = Path(*parent_parts[:part_count]) if part_count else Path(".")
        absolute_candidate = repository_path / candidate
        if _has_module_marker(absolute_candidate):
            return candidate
    return Path(".")


def _has_module_marker(path: Path) -> bool:
    return (path / MAVEN_MARKER_FILE_NAME).exists() or any((path / marker).exists() for marker in GRADLE_MARKER_FILE_NAMES)


def _detect_java_source_root(module_relative_path: Path) -> Path:
    parts = module_relative_path.parts
    for pattern in JAVA_SOURCE_ROOT_PATTERNS:
        pattern_length = len(pattern)
        if len(parts) >= pattern_length and tuple(part.lower() for part in parts[:pattern_length]) == pattern:
            return Path(*parts[:pattern_length])
    return Path(".")


def _detect_file_zone(module_relative_path: Path, *, source_root: Path) -> str:
    parts = {part.lower() for part in module_relative_path.parts}
    source_root_parts = {part.lower() for part in source_root.parts}
    if parts & GENERATED_PATH_PARTS or source_root_parts & GENERATED_PATH_PARTS:
        return GENERATED_ZONE
    if parts & VENDOR_PATH_PARTS:
        return "vendor"
    if parts & TOOLING_PATH_PARTS:
        return "tooling"
    if parts & TEST_PATH_PARTS:
        return "test"
    return DEFAULT_ZONE


def _module_zone(file_zone: str) -> Literal["prod", "test", "tooling", "vendor", "generated"]:
    if file_zone == "vendor":
        return "vendor"
    if file_zone == "tooling":
        return "tooling"
    return DEFAULT_ZONE


def _module_name(module_root: Path) -> str:
    if module_root == Path("."):
        return DEFAULT_MODULE_NAME
    return module_root.as_posix().replace("/", ":")


def _detect_module_ecosystem(repository_path: Path, module_root: Path) -> str:
    absolute_module_root = repository_path / module_root
    if (absolute_module_root / MAVEN_MARKER_FILE_NAME).exists():
        return DEFAULT_MODULE_ECOSYSTEM
    if any((absolute_module_root / marker).exists() for marker in GRADLE_MARKER_FILE_NAMES):
        return "gradle"
    return DEFAULT_MODULE_ECOSYSTEM
