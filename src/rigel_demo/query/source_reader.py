"""仓库源码切片读取。"""

from __future__ import annotations

from pathlib import Path

DEFAULT_SOURCE_SLICE_MAX_LINES = 120


class SourcePathError(ValueError):
    """源码路径不在当前仓库内。"""


class SourceLineRangeError(ValueError):
    """源码切片行号范围不可用。"""


class SourceFileNotFoundError(FileNotFoundError):
    """源码文件不存在。"""


class SourceReadError(RuntimeError):
    """源码文件读取失败。"""


class RepositorySourceReader:
    """从当前仓库安全读取源码切片。"""

    def __init__(self, repository_path: Path) -> None:
        self._repository_path = repository_path.resolve()

    def normalize_relative_path(self, relative_path: str) -> str:
        candidate_path = self._resolve_repository_file(relative_path)
        try:
            return candidate_path.relative_to(self._repository_path).as_posix()
        except ValueError as error:
            raise SourcePathError("源码路径必须位于当前仓库内") from error

    def read_slice(
        self,
        relative_path: str,
        *,
        start_line: int,
        end_line: int,
        max_lines: int = DEFAULT_SOURCE_SLICE_MAX_LINES,
    ) -> dict[str, object]:
        normalized_path = self.normalize_relative_path(relative_path)
        _ensure_positive_int(max_lines, "max_lines")
        _ensure_positive_int(start_line, "start_line")
        _ensure_positive_int(end_line, "end_line")
        if end_line < start_line:
            raise SourceLineRangeError("end_line 必须大于或等于 start_line")

        bounded_end_line = min(end_line, start_line + max_lines - 1)
        file_path = self._repository_path / normalized_path
        if not file_path.exists() or not file_path.is_file():
            raise SourceFileNotFoundError(f"未找到源码文件：{normalized_path}")

        try:
            lines = file_path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError as error:
            raise SourceReadError(f"源码文件不是 UTF-8 文本：{normalized_path}") from error
        except OSError as error:
            raise SourceReadError(f"读取源码文件失败：{normalized_path}") from error

        total_lines = len(lines)
        if start_line > total_lines:
            raise SourceLineRangeError(f"start_line 超出文件总行数：{total_lines}")

        actual_end_line = min(bounded_end_line, total_lines)
        selected_lines = lines[start_line - 1 : actual_end_line]
        return {
            "relative_path": normalized_path,
            "start_line": start_line,
            "end_line": actual_end_line,
            "requested_end_line": end_line,
            "truncated": actual_end_line < end_line,
            "total_lines": total_lines,
            "content": "\n".join(selected_lines),
        }

    def _resolve_repository_file(self, relative_path: str) -> Path:
        if not relative_path.strip():
            raise SourcePathError("源码路径不能为空")
        raw_path = Path(relative_path)
        if raw_path.is_absolute():
            candidate_path = raw_path.resolve()
        else:
            candidate_path = (self._repository_path / raw_path).resolve()
        try:
            # resolve 后再 relative_to，可以同时拦截绝对路径和 `..` 逃逸。
            candidate_path.relative_to(self._repository_path)
        except ValueError as error:
            raise SourcePathError("源码路径必须位于当前仓库内") from error
        return candidate_path


def _ensure_positive_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SourceLineRangeError(f"{name} 必须大于 0")
