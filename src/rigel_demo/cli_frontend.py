"""Rigel Web 前端构建。"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


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
