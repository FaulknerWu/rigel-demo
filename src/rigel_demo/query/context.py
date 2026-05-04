"""GraphRAG 上下文中的源码切片组装。"""

from __future__ import annotations

from collections.abc import Mapping

from rigel_demo.query.source_reader import (
    RepositorySourceReader,
    SourceFileNotFoundError,
    SourceLineRangeError,
    SourcePathError,
    SourceReadError,
)

DEFAULT_CONTEXT_SOURCE_SLICE_LIMIT = 1


def source_slices_for_anchors(
    anchors: list[dict[str, object]],
    *,
    source_reader: RepositorySourceReader,
) -> list[dict[str, object]]:
    source_slices: list[dict[str, object]] = []
    for anchor in anchors[:DEFAULT_CONTEXT_SOURCE_SLICE_LIMIT]:
        source_file_value = anchor.get("source_file")
        if not isinstance(source_file_value, Mapping):
            continue
        source_file = dict(source_file_value)
        relative_path = source_file.get("relative_path")
        if not isinstance(relative_path, str) or not relative_path.strip():
            continue
        try:
            source_slice = source_reader.read_slice(
                relative_path,
                start_line=int(anchor["start_line"]),
                end_line=int(anchor["end_line"]),
            )
        except (SourcePathError, SourceLineRangeError, SourceFileNotFoundError, SourceReadError):
            continue
        source_slices.append(
            {
                **source_slice,
                "anchor": {
                    "id": anchor["id"],
                    "role": anchor["role"],
                    "start_line": anchor["start_line"],
                    "start_col": anchor["start_col"],
                    "end_line": anchor["end_line"],
                    "end_col": anchor["end_col"],
                },
                "source_file": source_file,
            }
        )
    return source_slices
