from __future__ import annotations

import importlib
import importlib.util
import io
import logging
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from equity_analyst.pdf_writer import _log_weasyprint_render_failure, write_markdown_as_pdf


def _patch_fake_weasyprint_module(fake_html: MagicMock):
    """Avoid importing real WeasyPrint (native libs); ``write_markdown_as_pdf`` does ``from weasyprint import HTML``."""
    mod = ModuleType("weasyprint")
    mod.HTML = fake_html
    return patch.dict(sys.modules, {"weasyprint": mod})


def _weasyprint_importable() -> bool:
    if importlib.util.find_spec("weasyprint") is None:
        return False
    try:
        importlib.import_module("weasyprint")
    except Exception:
        return False
    return True


def _weasyprint_can_render_minimal_pdf() -> bool:
    """True only if WeasyPrint can produce bytes (import alone is not enough on some hosts)."""
    if not _weasyprint_importable():
        return False
    try:
        from weasyprint import HTML

        buf = io.BytesIO()
        HTML(string="<html><body>t</body></html>").write_pdf(target=buf)
        return buf.tell() > 50
    except Exception:
        return False


@pytest.mark.parametrize(
    ("exc", "must_contain", "must_not_contain"),
    [
        (ImportError("no module named 'cairo'"), "brew install pango cairo", "mismatched"),
        (
            AttributeError("'super' object has no attribute 'transform'"),
            "mismatched",
            "brew install pango cairo",
        ),
        (RuntimeError("boom"), "PDF skipped", "brew install pango cairo"),
    ],
)
def test_log_weasyprint_render_failure_routing(
    exc: BaseException,
    must_contain: str,
    must_not_contain: str,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    dest = tmp_path / "out.pdf"
    with caplog.at_level(logging.WARNING):
        _log_weasyprint_render_failure(dest, exc)
    joined = " ".join(r.message for r in caplog.records)
    assert must_contain in joined
    assert must_not_contain not in joined


@pytest.mark.skipif(
    not _weasyprint_can_render_minimal_pdf(),
    reason="weasyprint not available or cannot render PDF on this host",
)
def test_write_markdown_as_pdf_writes_non_empty_file(tmp_path: Path) -> None:
    md = (
        "# Title\n\n"
        "| a | b |\n|---|---|\n"
        "| 1 | 2 |\n\n"
        "```python\nx = 1\n```\n"
    )
    dest = tmp_path / "out.pdf"
    assert write_markdown_as_pdf(md, dest) is True
    assert dest.is_file()
    assert dest.stat().st_size > 500


def test_write_markdown_as_pdf_logs_native_hint_on_cairo_import_error(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    dest = tmp_path / "out.pdf"
    fake_html = MagicMock()
    fake_html.return_value.write_pdf.side_effect = ImportError("cannot import cairo")
    with _patch_fake_weasyprint_module(fake_html), caplog.at_level(logging.WARNING):
        assert write_markdown_as_pdf("# Hello", dest) is False
    joined = " ".join(r.message for r in caplog.records)
    assert "brew install pango cairo" in joined
    assert "mismatched" not in joined


def test_write_markdown_as_pdf_logs_dep_mismatch_on_attributeerror(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    dest = tmp_path / "out.pdf"
    fake_html = MagicMock()
    fake_html.return_value.write_pdf.side_effect = AttributeError(
        "'super' object has no attribute 'transform'"
    )
    with _patch_fake_weasyprint_module(fake_html), caplog.at_level(logging.WARNING):
        assert write_markdown_as_pdf("# Hello", dest) is False
    joined = " ".join(r.message for r in caplog.records)
    assert "mismatched" in joined
    assert "brew install pango cairo" not in joined


def test_write_markdown_as_pdf_logs_generic_on_other_render_error(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    dest = tmp_path / "out.pdf"
    fake_html = MagicMock()
    fake_html.return_value.write_pdf.side_effect = ValueError("bad pdf state")
    with _patch_fake_weasyprint_module(fake_html), caplog.at_level(logging.WARNING):
        assert write_markdown_as_pdf("# Hello", dest) is False
    joined = " ".join(r.message for r in caplog.records)
    assert "PDF skipped" in joined
    assert "brew install pango cairo" not in joined
