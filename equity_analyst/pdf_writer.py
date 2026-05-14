from __future__ import annotations

import logging
import re
from pathlib import Path

import markdown  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

# Pipe-table row boundary when the model glued multiple GFM rows on one line (3-column
# confidence-style tables: next row's first cell is like "2. Historical …").
_GLUE_PIPE_TABLE_ROW = re.compile(r"(?<=\|)\s+(?=\|\s*\d+\.\s+[A-Za-z(])")


def preprocess_markdown_for_pdf(markdown_text: str) -> str:
    """Normalize markdown so ``markdown.markdown(..., extensions=['tables'])`` yields real <table> blocks.

    Python-Markdown's table extension requires a blank line before a pipe table when the
    previous line is non-table prose (e.g. ``**Confidence Summary**`` immediately above
    ``| Section | ...``). Without it, the table is parsed as inline text inside one <p>.
    Also splits a few common glued multi-row pipe lines back onto separate lines.
    """
    lines = markdown_text.split("\n")
    expanded: list[str] = []
    in_fence = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            expanded.append(line)
            continue
        if in_fence:
            expanded.append(line)
            continue
        expanded.extend(_split_glued_pipe_table_rows_one_line(line).split("\n"))

    out: list[str] = []
    in_fence = False
    for line in expanded:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue
        if out and _needs_blank_line_before_pipe_table_row(line, out[-1]):
            out.append("")
        out.append(line)

    result = "\n".join(out)
    if markdown_text.endswith("\n") and not result.endswith("\n"):
        result += "\n"
    return result


def _is_probable_pipe_table_row(line: str) -> bool:
    s = line.strip()
    if not s.startswith("|"):
        return False
    return s.count("|") >= 2


def _is_table_separator_row(line: str) -> bool:
    s = line.strip()
    if not s.startswith("|") or "|" not in s[1:]:
        return False
    inner = [c.strip() for c in s.strip("|").split("|")]
    if not inner:
        return False
    for cell in inner:
        if not cell:
            return False
        if not re.fullmatch(r":?-{3,}:?", cell):
            return False
    return True


def _needs_blank_line_before_pipe_table_row(line: str, prev_line: str) -> bool:
    if not _is_probable_pipe_table_row(line):
        return False
    if not prev_line.strip():
        return False
    return not (
        _is_probable_pipe_table_row(prev_line) or _is_table_separator_row(prev_line)
    )


def _split_glued_pipe_table_rows_one_line(line: str) -> str:
    """If multiple 3-column pipe rows were emitted on one physical line, split at row boundaries."""
    if not _is_probable_pipe_table_row(line) or line.count("|") < 8:
        return line
    if not _GLUE_PIPE_TABLE_ROW.search(line):
        return line
    return _GLUE_PIPE_TABLE_ROW.sub("\n", line)


def _native_pdf_lib_hint_in_exception(exc: BaseException) -> bool:
    """True if the exception likely indicates missing cairo/pango/gdk-pixbuf (system) libraries."""
    msg = str(exc).lower()
    needles = ("cairo", "pango", "gdk-pixbuf", "gdk_pixbuf", "libgobject", "libffi", "cannot load library")
    return any(n in msg for n in needles)


def _log_weasyprint_render_failure(dest_path: Path, exc: BaseException) -> None:
    """Emit a targeted WARNING after ``HTML().write_pdf()`` failed."""
    exc_name = type(exc).__name__
    if isinstance(exc, (ImportError, OSError)) and _native_pdf_lib_hint_in_exception(exc):
        logger.warning(
            "PDF generation failed for %s (%s: %s). On macOS try: "
            "``brew install pango cairo gdk-pixbuf libffi``. Run continues.",
            dest_path,
            exc_name,
            exc,
        )
    elif isinstance(exc, (AttributeError, TypeError)):
        logger.warning(
            "PDF generation failed for %s (%s: %s). WeasyPrint Python deps appear mismatched; "
            "try ``.venv/bin/pip install -U weasyprint pydyf fonttools``. Run continues.",
            dest_path,
            exc_name,
            exc,
        )
    else:
        logger.warning(
            "WeasyPrint rendering failed for %s: %s: %s. PDF skipped. Run continues.",
            dest_path,
            exc_name,
            exc,
        )

_PDF_CSS = """
@page { margin: 1in; }
html {
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  font-size: 11pt;
  line-height: 1.45;
  color: #1a1a1a;
}
body { margin: 0; }
h1 { font-size: 1.6em; margin-top: 0.6em; margin-bottom: 0.35em; font-weight: 600; }
h2 { font-size: 1.35em; margin-top: 0.75em; margin-bottom: 0.35em; font-weight: 600; }
h3 { font-size: 1.15em; margin-top: 0.65em; margin-bottom: 0.3em; font-weight: 600; }
h4, h5, h6 { font-size: 1.05em; margin-top: 0.5em; margin-bottom: 0.25em; font-weight: 600; }
p { margin: 0.45em 0; }
ul, ol { margin: 0.45em 0; padding-left: 1.35em; }
li { margin: 0.2em 0; }
blockquote {
  margin: 0.6em 0;
  padding: 0.35em 0.85em;
  border-left: 3px solid #bbb;
  color: #333;
  background: #f7f7f7;
}
pre {
  background: #f0f0f0;
  border: 1px solid #ddd;
  border-radius: 4px;
  padding: 0.65em 0.85em;
  overflow-x: auto;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 0.92em;
  line-height: 1.35;
}
code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 0.92em;
  background: #f0f0f0;
  padding: 0.12em 0.35em;
  border-radius: 3px;
}
pre code { background: none; padding: 0; border-radius: 0; }
table { border-collapse: collapse; width: 100%; margin: 0.75em 0; font-size: 0.98em; }
th, td { border: 1px solid #bbb; padding: 0.35em 0.55em; text-align: left; vertical-align: top; }
th { background: #eaeaea; font-weight: 600; }
hr { border: none; border-top: 1px solid #ccc; margin: 1em 0; }
a { color: #0b57d0; }
"""


def write_markdown_as_pdf(markdown_text: str, dest_path: Path) -> bool:
    """Render markdown to PDF at ``dest_path``. Returns True on success, False on graceful failure."""
    try:
        from weasyprint import HTML  # type: ignore[import-untyped]
    except Exception as exc:
        logger.warning(
            "PDF generation skipped: WeasyPrint could not be loaded (%s: %s). "
            "Install Python deps (``pip install weasyprint``) and on macOS system libraries, e.g.: "
            "``brew install pango cairo gdk-pixbuf libffi``. Run continues without PDF.",
            type(exc).__name__,
            exc,
        )
        return False

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        body_html = markdown.markdown(
            preprocess_markdown_for_pdf(markdown_text),
            extensions=["fenced_code", "tables", "toc"],
        )
    except Exception as exc:
        logger.warning(
            "PDF generation skipped: markdown parse failed (%s: %s).",
            type(exc).__name__,
            exc,
        )
        return False

    doc = (
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\"/>"
        f"<style>{_PDF_CSS}</style></head><body>{body_html}</body></html>"
    )
    base_url = dest_path.parent.resolve().as_uri() + "/"
    try:
        HTML(string=doc, base_url=base_url).write_pdf(str(dest_path))
    except Exception as exc:
        _log_weasyprint_render_failure(dest_path, exc)
        return False

    try:
        if not dest_path.is_file() or dest_path.stat().st_size == 0:
            logger.warning("PDF generation produced empty or missing file at %s.", dest_path)
            return False
    except OSError as exc:
        logger.warning("Could not stat PDF output %s: %s", dest_path, exc)
        return False

    return True


def maybe_write_pdf_sibling(*, pdf_output_enabled: bool, md_path: Path, markdown_text: str) -> None:
    """If enabled, write ``md_path`` with the same basename and ``.pdf`` extension."""
    if not pdf_output_enabled:
        return
    pdf_path = md_path.with_suffix(".pdf")
    try:
        ok = write_markdown_as_pdf(markdown_text, pdf_path)
    except Exception as exc:
        logger.warning(
            "Unexpected error during PDF generation for %s (%s: %s). Run continues.",
            md_path,
            type(exc).__name__,
            exc,
        )
        return
    if ok:
        logger.info("Wrote PDF %s", pdf_path)
