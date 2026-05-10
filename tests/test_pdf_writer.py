from __future__ import annotations

import importlib
import importlib.util
import io
from pathlib import Path

import pytest

from equity_analyst.pdf_writer import write_markdown_as_pdf


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
