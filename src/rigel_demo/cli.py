"""Rigel 本地命令行入口。"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import webbrowser
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

from rigel_demo.llm.config import DEFAULT_CHAT_SYSTEM_PROMPT, DEFAULT_SUMMARY_SYSTEM_PROMPT


RIGEL_WORKSPACE_DIRECTORY_NAME = ".rigel"
RIGEL_CONFIG_FILE_NAME = "config.json"
FALKORDB_DATABASE_FILE_NAME = "falkordb.db"
WORKSPACE_STATE_FILE_NAME = "rigel.json"
WEB_STATIC_DIRECTORY_NAME = "web/static"
DEFAULT_GRAPH_NAME = "rigel"
WEB_CONFIG_SECTION_NAME = "web"
DEFAULT_CONFIG_DOCUMENT = {
    "web": {
        "host": "127.0.0.1",
        "port": 5000,
        "open_browser": True,
    },
    "chat": {
        "provider": "openai",
        "model": "gpt-5.2",
        "api_key": "sk-your-openai-key",
        "base_url": None,
        "timeout_seconds": 60,
        "temperature": None,
        "max_output_tokens": None,
        "system_prompt": DEFAULT_CHAT_SYSTEM_PROMPT,
    },
    "summary": {
        "provider": "openai",
        "model": "gpt-5.2",
        "api_key": "sk-your-openai-key",
        "base_url": None,
        "timeout_seconds": 60,
        "temperature": 0,
        "max_output_tokens": 300,
        "system_prompt": DEFAULT_SUMMARY_SYSTEM_PROMPT,
    },
    "embedding": {
        "provider": "openai",
        "format": "openai_embeddings",
        "model": "text-embedding-3-small",
        "api_key": "sk-your-openai-key",
        "base_url": None,
        "dimensions": 512,
        "timeout_seconds": 60,
        "batch_size": 64,
    },
}


@dataclass(frozen=True, slots=True)
class InitResult:
    """初始化命令的执行结果。"""

    repository_path: str
    workspace_path: str
    config_path: str
    config_created: bool


@dataclass(frozen=True, slots=True)
class IndexResult:
    """索引命令的执行结果。"""

    repository_path: str
    workspace_path: str
    database_path: str
    state_path: str
    graph_name: str
    indexed_file_count: int
    graph_node_count: int
    graph_edge_count: int
    index_mode: str = "full"
    added_files: tuple[str, ...] = ()
    modified_files: tuple[str, ...] = ()
    deleted_files: tuple[str, ...] = ()
    skipped_file_count: int = 0
    deleted_node_count: int = 0
    duration_ms: int = 0


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


def main(argv: Sequence[str] | None = None) -> int:
    """执行 Rigel CLI。"""

    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.command_handler(args)


def init_repository(repository_path: Path | None = None) -> InitResult:
    """在目标仓库创建 Rigel 本地工作目录。"""

    resolved_repository_path = (repository_path or Path.cwd()).resolve()
    workspace_path = resolved_repository_path / RIGEL_WORKSPACE_DIRECTORY_NAME
    config_path = workspace_path / RIGEL_CONFIG_FILE_NAME

    workspace_path.mkdir(parents=True, exist_ok=True)
    config_created = _write_default_config_if_missing(config_path)

    return InitResult(
        repository_path=str(resolved_repository_path),
        workspace_path=str(workspace_path),
        config_path=str(config_path),
        config_created=config_created,
    )


def index_repository_workspace(repository_path: Path | None = None, *, incremental: bool = False) -> IndexResult:
    """扫描目标仓库并重建 Rigel 本地图数据库。"""

    from rigel_demo.storage.falkordb_store import FalkorDBConfig, FalkorDBStore
    from rigel_demo.indexing.repository_indexer import index_repository, index_repository_incremental

    resolved_repository_path = (repository_path or Path.cwd()).resolve()
    workspace_path = resolved_repository_path / RIGEL_WORKSPACE_DIRECTORY_NAME
    config_path = workspace_path / RIGEL_CONFIG_FILE_NAME
    database_path = workspace_path / FALKORDB_DATABASE_FILE_NAME
    state_path = workspace_path / WORKSPACE_STATE_FILE_NAME

    if not config_path.exists():
        raise FileNotFoundError(f"未找到配置文件，请先执行 rigel init 并填写配置：{config_path}")

    workspace_path.mkdir(parents=True, exist_ok=True)
    started_at = perf_counter()

    if incremental:
        if not database_artifact_exists(database_path):
            raise FileNotFoundError(f"未找到可增量索引的图数据库，请先执行 rigel index：{database_path}")
        # 增量模式复用旧数据库中的文件哈希，先删旧子图再写新子图，保持演示实现简单可观察。
        store = FalkorDBStore.connect(
            FalkorDBConfig(
                graph_name=DEFAULT_GRAPH_NAME,
                database_path=str(database_path),
            )
        )
        index_result = index_repository_incremental(
            resolved_repository_path,
            previous_file_hashes=store.list_java_file_hashes(),
        )
        deleted_node_count = store.delete_file_subgraphs(
            [*index_result.modified_files, *index_result.deleted_files]
        )
        if index_result.graph.nodes or index_result.graph.edges:
            store.upsert_graph(index_result.graph)
        graph_node_count, graph_edge_count = store.graph_counts()
        result = IndexResult(
            repository_path=str(resolved_repository_path),
            workspace_path=str(workspace_path),
            database_path=str(database_path),
            state_path=str(state_path),
            graph_name=DEFAULT_GRAPH_NAME,
            indexed_file_count=index_result.indexed_file_count,
            graph_node_count=graph_node_count,
            graph_edge_count=graph_edge_count,
            index_mode="incremental",
            added_files=tuple(index_result.added_files),
            modified_files=tuple(index_result.modified_files),
            deleted_files=tuple(index_result.deleted_files),
            skipped_file_count=len(index_result.skipped_files),
            deleted_node_count=deleted_node_count,
            duration_ms=_duration_ms(started_at),
        )
        _write_workspace_state(state_path, result)
        return result

    _remove_database_artifacts(database_path)
    index_result = index_repository(resolved_repository_path)
    FalkorDBStore.connect(
        FalkorDBConfig(
            graph_name=DEFAULT_GRAPH_NAME,
            database_path=str(database_path),
        )
    ).upsert_graph(index_result.graph)

    result = IndexResult(
        repository_path=str(resolved_repository_path),
        workspace_path=str(workspace_path),
        database_path=str(database_path),
        state_path=str(state_path),
        graph_name=DEFAULT_GRAPH_NAME,
        indexed_file_count=index_result.indexed_file_count,
        graph_node_count=len(index_result.graph.nodes),
        graph_edge_count=len(index_result.graph.edges),
        index_mode="full",
        duration_ms=_duration_ms(started_at),
    )
    _write_workspace_state(state_path, result)
    return result


def _build_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数解析器。"""

    parser = argparse.ArgumentParser(
        prog="rigel",
        description="Rigel 本地代码图谱工具。",
    )
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    init_parser = subparsers.add_parser(
        "init",
        help="在当前仓库初始化 Rigel 本地工作目录。",
        description="在当前仓库初始化 .rigel 目录并生成默认配置文件。",
    )
    init_parser.set_defaults(command_handler=_handle_init_command)

    index_parser = subparsers.add_parser(
        "index",
        help="为当前仓库生成或重建 Rigel 本地图数据库。",
        description="读取当前目录 .rigel/config.json，扫描源码并重建 .rigel/falkordb.db。",
    )
    index_parser.add_argument(
        "--incremental",
        action="store_true",
        help="基于已有图数据库执行演示级增量索引。",
    )
    index_parser.set_defaults(command_handler=_handle_index_command)

    web_parser = subparsers.add_parser(
        "web",
        help="启动当前仓库的 Rigel Web 演示后端。",
        description="读取当前目录 .rigel/config.json 与 .rigel/falkordb.db，启动 Web 演示后端。",
    )
    web_parser.set_defaults(command_handler=_handle_web_command)

    return parser


def _handle_init_command(_args: argparse.Namespace) -> int:
    """处理 init 命令。"""

    result = init_repository()
    print("Rigel 初始化完成")
    print(f"仓库目录: {result.repository_path}")
    print(f"工作目录: {result.workspace_path}")
    print(f"配置文件: {result.config_path}")
    print(f"配置状态: {'已生成' if result.config_created else '已存在，未覆盖'}")
    print("请填写配置文件后执行 rigel index。")
    return 0


def _handle_index_command(_args: argparse.Namespace) -> int:
    """处理 index 命令。"""

    try:
        result = index_repository_workspace(incremental=_args.incremental)
    except FileNotFoundError as error:
        print(str(error))
        return 1

    print("Rigel 索引完成")
    print(f"仓库目录: {result.repository_path}")
    print(f"工作目录: {result.workspace_path}")
    print(f"图数据库: {result.database_path}")
    print(f"索引模式: {_index_mode_label(result)}")
    print(f"索引文件: {result.indexed_file_count}")
    print(f"图谱节点: {result.graph_node_count}")
    print(f"图谱边: {result.graph_edge_count}")
    print(f"耗时: {result.duration_ms} ms")
    if result.index_mode == "incremental":
        print(f"新增文件: {len(result.added_files)}")
        print(f"修改文件: {len(result.modified_files)}")
        print(f"删除文件: {len(result.deleted_files)}")
        print(f"跳过文件: {result.skipped_file_count}")
        print(f"删除旧节点: {result.deleted_node_count}")
        for label, files in (
            ("新增", result.added_files),
            ("修改", result.modified_files),
            ("删除", result.deleted_files),
        ):
            for relative_path in files:
                print(f"{label}: {relative_path}")
    return 0


def _handle_web_command(_args: argparse.Namespace) -> int:
    """处理 web 命令。"""

    import uvicorn

    from rigel_demo.web.app import create_app

    repository_path = Path.cwd().resolve()
    workspace_path = repository_path / RIGEL_WORKSPACE_DIRECTORY_NAME
    database_path = repository_path / RIGEL_WORKSPACE_DIRECTORY_NAME / FALKORDB_DATABASE_FILE_NAME
    try:
        web_config = WebConfig.from_repository(repository_path)
    except (FileNotFoundError, WebConfigurationError) as error:
        print(str(error))
        return 1

    if not database_artifact_exists(database_path):
        print(f"未找到图数据库: {database_path}")
        print("请先在目标仓库执行 rigel index。")
        return 1

    frontend_result = build_frontend(workspace_path / WEB_STATIC_DIRECTORY_NAME)
    if not frontend_result.success:
        print(frontend_result.message)
        return 1

    url = f"http://{web_config.host}:{web_config.port}"
    print("Rigel Web 演示后端已启动", flush=True)
    print(f"仓库目录: {repository_path}", flush=True)
    print(f"图数据库: {database_path}", flush=True)
    print(f"前端目录: {frontend_result.static_path}", flush=True)
    print(f"访问地址: {url}", flush=True)
    if web_config.open_browser:
        # 浏览器打开失败不影响后端启动；webbrowser 会按当前系统可用性自行处理。
        webbrowser.open(url)

    uvicorn.run(create_app(repository_path), host=web_config.host, port=web_config.port)
    return 0


@dataclass(frozen=True, slots=True)
class FrontendBuildResult:
    """前端构建结果。"""

    success: bool
    static_path: Path
    message: str = ""


def build_frontend(static_path: Path) -> FrontendBuildResult:
    """把内置前端构建到当前仓库 `.rigel` 工作目录。"""

    frontend_path = Path(__file__).resolve().parent / "web" / "frontend"
    package_json_path = frontend_path / "package.json"
    if not package_json_path.exists():
        return FrontendBuildResult(
            success=False,
            static_path=static_path,
            message=f"未找到前端源码目录: {frontend_path}",
        )

    npm_path = shutil.which("npm")
    if npm_path is None:
        return FrontendBuildResult(
            success=False,
            static_path=static_path,
            message="未找到 npm，请先安装 Node.js/npm，并确认 npm --version 可用后再构建 Rigel Web 前端。",
        )

    node_modules_path = frontend_path / "node_modules"
    if not node_modules_path.exists():
        install_result = subprocess.run(
            [npm_path, "install"],
            cwd=frontend_path,
            check=False,
            capture_output=True,
            text=True,
        )
        if install_result.returncode != 0:
            return FrontendBuildResult(
                success=False,
                static_path=static_path,
                message=_frontend_command_error("前端依赖安装失败", install_result),
            )

    build_result = subprocess.run(
        [npm_path, "run", "build", "--", "--outDir", str(static_path), "--emptyOutDir"],
        cwd=frontend_path,
        check=False,
        capture_output=True,
        text=True,
    )
    if build_result.returncode != 0:
        return FrontendBuildResult(
            success=False,
            static_path=static_path,
            message=_frontend_command_error("前端构建失败", build_result),
        )

    return FrontendBuildResult(success=True, static_path=static_path)


def _frontend_command_error(title: str, result: subprocess.CompletedProcess[str]) -> str:
    output = "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)
    if not output:
        return title
    return f"{title}:\n{output}"


def _write_default_config_if_missing(config_path: Path) -> bool:
    """只在缺失时写入默认配置，避免覆盖用户已填写的密钥与模型。"""

    if config_path.exists():
        return False

    config_path.write_text(
        json.dumps(DEFAULT_CONFIG_DOCUMENT, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return True


def _write_workspace_state(state_path: Path, result: IndexResult) -> None:
    """写入 MCP 与后续命令可复用的索引状态文件。"""

    state = {
        **asdict(result),
        "indexed_at": datetime.now(UTC).isoformat(),
        "last_index_mode": result.index_mode,
        "last_index_result": index_result_payload(result),
        "last_incremental_result": index_result_payload(result) if result.index_mode == "incremental" else None,
    }
    state_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def index_result_payload(result: IndexResult) -> dict[str, object]:
    """把索引结果转换成 CLI 和 Web 可共用的展示载荷。"""

    return {
        "mode": result.index_mode,
        "mode_label": _index_mode_label(result),
        "added_files": list(result.added_files),
        "modified_files": list(result.modified_files),
        "deleted_files": list(result.deleted_files),
        "skipped_file_count": result.skipped_file_count,
        "indexed_file_count": result.indexed_file_count,
        "deleted_node_count": result.deleted_node_count,
        "graph_node_count": result.graph_node_count,
        "graph_edge_count": result.graph_edge_count,
        "duration_ms": result.duration_ms,
    }


def _index_mode_label(result: IndexResult) -> str:
    if result.index_mode == "incremental":
        return "增量"
    return "全量"


def _duration_ms(started_at: float) -> int:
    return int((perf_counter() - started_at) * 1000)


def database_artifact_exists(database_path: Path) -> bool:
    """判断 FalkorDBLite 是否已为指定数据库路径生成运行时产物。"""

    return database_path.exists() or database_path.with_name(f"{database_path.name}.settings").exists()


def _remove_database_artifacts(database_path: Path) -> None:
    """重建索引前清理 FalkorDBLite 为同一路径生成的旧产物。"""

    database_path.unlink(missing_ok=True)
    database_path.with_name(f"{database_path.name}.settings").unlink(missing_ok=True)


def _read_web_config(repository_path: Path) -> dict[str, Any]:
    """读取 `.rigel/config.json` 中的 Web 配置段。"""

    config_path = repository_path / RIGEL_WORKSPACE_DIRECTORY_NAME / RIGEL_CONFIG_FILE_NAME
    if not config_path.exists():
        raise FileNotFoundError(f"未找到配置文件，请先执行 rigel init 并填写配置：{config_path}")

    try:
        config_document = json.loads(config_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise WebConfigurationError(f"读取 Web 配置文件失败：{config_path}") from error
    except json.JSONDecodeError as error:
        raise WebConfigurationError(f"Web 配置文件不是合法 JSON：{config_path}") from error

    if not isinstance(config_document, dict):
        raise WebConfigurationError("Web 配置文件根节点必须是 JSON 对象")

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


if __name__ == "__main__":
    raise SystemExit(main())
