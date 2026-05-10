from __future__ import annotations

import logging
from pathlib import Path

import markdown  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

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
        from weasyprint import HTML  # type: ignore[import-not-found]
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
            markdown_text,
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
        logger.warning(
            "PDF generation failed for %s (%s: %s). On macOS try: "
            "``brew install pango cairo gdk-pixbuf libffi``. Run continues.",
            dest_path,
            type(exc).__name__,
            exc,
        )
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
