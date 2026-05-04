from __future__ import annotations

from unittest import TestCase

from rigel_demo.entities import Anchor, File, Module, Repository, Summary


class PhysicalNodeContractTest(TestCase):
    def test_repository_requires_non_empty_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "Repository.repo_id"):
            Repository(repo_id="", name="demo")

        with self.assertRaisesRegex(ValueError, "Repository.name"):
            Repository(repo_id="repo:demo", name=" ")

    def test_module_requires_non_empty_fields(self) -> None:
        invalid_cases = [
            {"module_id": ""},
            {"name": ""},
            {"root_path": ""},
            {"ecosystem": ""},
            {"zone": ""},
        ]

        for invalid_case in invalid_cases:
            with self.subTest(invalid_case=invalid_case):
                with self.assertRaisesRegex(ValueError, "Module"):
                    _module(**invalid_case)

    def test_file_requires_non_empty_fields(self) -> None:
        invalid_cases = [
            {"file_id": ""},
            {"relative_path": ""},
            {"language": ""},
            {"zone": ""},
            {"content_hash": ""},
            {"position_encoding": ""},
        ]

        for invalid_case in invalid_cases:
            with self.subTest(invalid_case=invalid_case):
                with self.assertRaisesRegex(ValueError, "File"):
                    _file(**invalid_case)


class AnchorContractTest(TestCase):
    def test_anchor_requires_positive_line_and_column_values(self) -> None:
        invalid_cases = [
            {"start_line": 0},
            {"start_col": 0},
            {"end_line": 0},
            {"end_col": 0},
            {"start_line": True},
        ]

        for invalid_case in invalid_cases:
            with self.subTest(invalid_case=invalid_case):
                with self.assertRaisesRegex(ValueError, "Anchor"):
                    _anchor(**invalid_case)

    def test_anchor_requires_non_empty_id_and_role(self) -> None:
        with self.assertRaisesRegex(ValueError, "Anchor.anchor_id"):
            _anchor(anchor_id=" ")

        with self.assertRaisesRegex(ValueError, "Anchor.role"):
            _anchor(role="")

    def test_anchor_rejects_end_position_before_start_position(self) -> None:
        with self.assertRaisesRegex(ValueError, "结束位置"):
            _anchor(start_line=3, start_col=10, end_line=3, end_col=2)


class SummaryContractTest(TestCase):
    def test_summary_requires_embedding_dimensions_to_match_vector(self) -> None:
        with self.assertRaisesRegex(ValueError, "embedding"):
            _summary(embedding_dimensions=3, embedding=[1.0, 0.0])

    def test_summary_rejects_non_numeric_embedding_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "全部是数字"):
            _summary(embedding=[1.0, "0.0"])

    def test_summary_rejects_invalid_purpose(self) -> None:
        with self.assertRaisesRegex(ValueError, "purpose"):
            _summary(purpose="search")


def _summary(
    *,
    purpose: str = "retrieval",
    embedding_dimensions: int = 2,
    embedding: list[object] | None = None,
) -> Summary:
    return Summary(
        summary_id="summary:demo:retrieval",
        text="PaymentService 处理付款流程",
        purpose=purpose,
        source_hash="sha256:source",
        summary_model="summary-model",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=embedding_dimensions,
        embedding=embedding or [1.0, 0.0],
    )


def _module(
    *,
    module_id: str = "module:demo:root",
    name: str = "root",
    root_path: str = ".",
    ecosystem: str = "maven",
    zone: str = "prod",
) -> Module:
    return Module(
        module_id=module_id,
        name=name,
        root_path=root_path,
        ecosystem=ecosystem,
        zone=zone,
    )


def _file(
    *,
    file_id: str = "file:demo:src/main/java/demo/PaymentService.java",
    relative_path: str = "src/main/java/demo/PaymentService.java",
    language: str = "java",
    zone: str = "prod",
    content_hash: str = "sha256:file",
    position_encoding: str = "UTF-8",
) -> File:
    return File(
        file_id=file_id,
        relative_path=relative_path,
        language=language,
        zone=zone,
        content_hash=content_hash,
        position_encoding=position_encoding,
    )


def _anchor(
    *,
    anchor_id: str = "anchor:entity:demo:PaymentService:definition",
    start_line: object = 1,
    start_col: object = 1,
    end_line: object = 1,
    end_col: object = 10,
    role: str = "definition",
) -> Anchor:
    return Anchor(
        anchor_id=anchor_id,
        start_line=start_line,
        start_col=start_col,
        end_line=end_line,
        end_col=end_col,
        role=role,
    )
