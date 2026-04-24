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
from typing import Sequence


RIGEL_WORKSPACE_DIRECTORY_NAME = ".rigel"
FALKORDB_DATABASE_FILE_NAME = "falkordb.db"
WORKSPACE_STATE_FILE_NAME = "rigel.json"
WEB_STATIC_DIRECTORY_NAME = "web/static"
DEFAULT_GRAPH_NAME = "rigel"


@dataclass(frozen=True, slots=True)
class InitResult:
    """初始化命令的执行结果。"""

    repository_path: str
    workspace_path: str
    database_path: str
    state_path: str
    graph_name: str
    indexed_file_count: int
    graph_node_count: int
    graph_edge_count: int


def main(argv: Sequence[str] | None = None) -> int:
    """执行 Rigel CLI。"""

    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.command_handler(args)


def init_repository(repository_path: Path | None = None) -> InitResult:
    """在目标仓库创建 Rigel 本地工作目录。"""

    from rigel_demo.storage.falkordb_store import FalkorDBConfig, FalkorDBStore
    from rigel_demo.indexing.repository_indexer import index_repository

    resolved_repository_path = (repository_path or Path.cwd()).resolve()
    workspace_path = resolved_repository_path / RIGEL_WORKSPACE_DIRECTORY_NAME
    database_path = workspace_path / FALKORDB_DATABASE_FILE_NAME
    state_path = workspace_path / WORKSPACE_STATE_FILE_NAME

    workspace_path.mkdir(parents=True, exist_ok=True)
    database_path.unlink(missing_ok=True)
    index_result = index_repository(resolved_repository_path)
    FalkorDBStore.connect(
        FalkorDBConfig(
            graph_name=DEFAULT_GRAPH_NAME,
            database_path=str(database_path),
        )
    ).upsert_graph(index_result.graph)

    result = InitResult(
        repository_path=str(resolved_repository_path),
        workspace_path=str(workspace_path),
        database_path=str(database_path),
        state_path=str(state_path),
        graph_name=DEFAULT_GRAPH_NAME,
        indexed_file_count=index_result.indexed_file_count,
        graph_node_count=len(index_result.graph.nodes),
        graph_edge_count=len(index_result.graph.edges),
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
        description="在当前仓库初始化 .rigel 目录和本地图数据库文件。",
    )
    init_parser.set_defaults(command_handler=_handle_init_command)

    web_parser = subparsers.add_parser(
        "web",
        help="启动当前仓库的 Rigel Web 演示后端。",
        description="读取当前目录 .rigel/falkordb.db，启动 Web 演示后端并打开浏览器。",
    )
    web_parser.add_argument("--host", default="127.0.0.1", help="Web 服务监听地址。")
    web_parser.add_argument("--port", default=5000, type=int, help="Web 服务监听端口。")
    web_parser.add_argument("--no-open", action="store_true", help="只启动服务，不自动打开浏览器。")
    web_parser.set_defaults(command_handler=_handle_web_command)

    return parser


def _handle_init_command(_args: argparse.Namespace) -> int:
    """处理 init 命令。"""

    result = init_repository()
    print("Rigel 初始化完成")
    print(f"仓库目录: {result.repository_path}")
    print(f"工作目录: {result.workspace_path}")
    print(f"图数据库: {result.database_path}")
    print(f"索引文件: {result.indexed_file_count}")
    print(f"图谱节点: {result.graph_node_count}")
    print(f"图谱边: {result.graph_edge_count}")
    return 0


def _handle_web_command(args: argparse.Namespace) -> int:
    """处理 web 命令。"""

    import uvicorn

    from rigel_demo.web.app import create_app

    repository_path = Path.cwd().resolve()
    workspace_path = repository_path / RIGEL_WORKSPACE_DIRECTORY_NAME
    database_path = repository_path / RIGEL_WORKSPACE_DIRECTORY_NAME / FALKORDB_DATABASE_FILE_NAME
    if not database_path.exists():
        print(f"未找到图数据库: {database_path}")
        print("请先在目标仓库执行 rigel init。")
        return 1

    frontend_result = build_frontend(workspace_path / WEB_STATIC_DIRECTORY_NAME)
    if not frontend_result.success:
        print(frontend_result.message)
        return 1

    url = f"http://{args.host}:{args.port}"
    print("Rigel Web 演示后端已启动", flush=True)
    print(f"仓库目录: {repository_path}", flush=True)
    print(f"图数据库: {database_path}", flush=True)
    print(f"前端目录: {frontend_result.static_path}", flush=True)
    print(f"访问地址: {url}", flush=True)
    if not args.no_open:
        webbrowser.open(url)

    uvicorn.run(create_app(repository_path), host=args.host, port=args.port)
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
            message="未找到 npm，无法构建 Rigel Web 前端。",
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


def _write_workspace_state(state_path: Path, result: InitResult) -> None:
    """写入 MCP 与后续命令可复用的本地状态文件。"""

    state = {
        **asdict(result),
        "initialized_at": datetime.now(UTC).isoformat(),
    }
    state_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
