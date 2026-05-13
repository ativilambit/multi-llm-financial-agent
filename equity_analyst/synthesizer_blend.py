from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

# Horizon blend rows are always **qual : quant** (qualitative first, quantitative second).

T0BlendPreset = Literal["default", "quant_lean", "quant_dominant", "qual_dominant"]

T0_BLEND_QUAL_QUANT: dict[T0BlendPreset, tuple[int, int]] = {
    "default": (49, 51),
    "quant_lean": (40, 60),
    "quant_dominant": (1, 99),
    "qual_dominant": (99, 1),
}

SYNTHESIZER_T0_BLEND_PLACEHOLDER = "__T0_BLEND_LITERAL__"


def format_t0_blend_qual_quant_literal(preset: T0BlendPreset) -> str:
    """Canonical ``qual : quant`` spacing for the T-0 rows (qualitative first)."""
    q, u = T0_BLEND_QUAL_QUANT[preset]
    return f"{q} : {u}"


def inject_t0_blend_into_synthesizer_system_prompt(raw: str, preset: T0BlendPreset) -> str:
    """Substitute T-0 fenced-table placeholders for the resolved preset literal."""
    lit = format_t0_blend_qual_quant_literal(preset)
    return raw.replace(SYNTHESIZER_T0_BLEND_PLACEHOLDER, lit)


def normalize_t0_blend_preset(v: object) -> T0BlendPreset:
    if v is None:
        return "default"
    s = str(v).strip().lower()
    if s == "default":
        return "default"
    if s == "quant_lean":
        return "quant_lean"
    if s == "quant_dominant":
        return "quant_dominant"
    if s == "qual_dominant":
        return "qual_dominant"
    raise ValueError(
        "t0_blend_preset must be one of: default, quant_lean, quant_dominant, qual_dominant "
        f"(got {v!r})",
    )


_RE_BLEND_49_51 = re.compile(r"\b49\s*:\s*51\b")
_RE_BLEND_51_49 = re.compile(r"\b51\s*:\s*49\b")
_RE_BLEND_55_45 = re.compile(r"\b55\s*:\s*45\b")
_RE_BLEND_45_55 = re.compile(r"\b45\s*:\s*55\b")
_RE_BLEND_40_60 = re.compile(r"\b40\s*:\s*60\b")
_RE_BLEND_60_40 = re.compile(r"\b60\s*:\s*40\b")
_RE_BLEND_1_99 = re.compile(r"\b1\s*:\s*99\b")
_RE_BLEND_99_1 = re.compile(r"\b99\s*:\s*1\b")
_RE_BAD_QUANT_FIRST_49_51 = re.compile(r"49\s+Quant\s*:\s*51\s+Qual", re.IGNORECASE)
_RE_BAD_QUANT_COLON_QUAL_LABEL = re.compile(r"\bquant\s*:\s*qual\b", re.IGNORECASE)
_RE_BAD_QUAL_COLON_QUANT_WORDS = re.compile(r"\bqualitative\s*:\s*quantitative\b", re.IGNORECASE)
_RE_BAD_PCT_ORDER_49_51_ROW = re.compile(
    r"51\s*%\s*qualitative.{0,160}49\s*%\s*quantitative",
    re.IGNORECASE | re.DOTALL,
)
_RE_BAD_PCT_ORDER_55_45_ROW = re.compile(
    r"45\s*%\s*qualitative.{0,160}55\s*%\s*quantitative",
    re.IGNORECASE | re.DOTALL,
)
_RE_BAD_PCT_ORDER_55_45_ROW_ALT = re.compile(
    r"55\s*%\s*quantitative.{0,160}45\s*%\s*qualitative",
    re.IGNORECASE | re.DOTALL,
)


def _markdown_row_cells(line: str) -> list[str] | None:
    s = line.strip()
    if not s.startswith("|"):
        return None
    parts = [p.strip() for p in s.split("|")]
    cells = [p for p in parts if p]
    return cells or None


def _is_t0_horizon_label(cell: str) -> bool:
    minus = "\N{MINUS SIGN}"
    c = cell.lower().replace(minus, "-")
    if "t+1" in c or "t + 1" in c:
        return False
    if "t-3" in c and "t-1" in c:
        return False
    if "t-0" in c or re.search(r"\bt0\b", c):
        return True
    if "pre-open" in c and "event day" in c:
        return True
    if "same-day intraday" in c or "same day intraday" in c:
        return True
    return "mid-day" in c or "post-print" in c or "post-amc" in c


def _parse_qual_quant_pair(cell: str) -> tuple[int, int] | None:
    m = re.search(r"\b(\d{1,3})\s*:\s*(\d{1,3})\b", cell)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def iter_t0_row_blend_pairs(synthesis_text: str) -> list[tuple[int, int]]:
    """Extract ``(qual, quant)`` digit pairs from markdown table rows whose horizon cell is T-0."""
    out: list[tuple[int, int]] = []
    for line in synthesis_text.splitlines():
        cells = _markdown_row_cells(line)
        if not cells or len(cells) < 2:
            continue
        if not _is_t0_horizon_label(cells[0]):
            continue
        pair = _parse_qual_quant_pair(cells[1])
        if pair is not None:
            out.append(pair)
    return out


def _t0_preset_table_followups(synthesis_text: str, *, t0_blend_preset: T0BlendPreset) -> list[str]:
    allowed = T0_BLEND_QUAL_QUANT[t0_blend_preset]
    wrong_others = {p for k, p in T0_BLEND_QUAL_QUANT.items() if k != t0_blend_preset}
    rows = iter_t0_row_blend_pairs(synthesis_text)
    if not rows:
        return []
    out: list[str] = []
    if len(set(rows)) > 1:
        out.append(
            "Section 8 horizon blend: multiple different T-0 blend digit pairs in markdown tables; "
            "use one consistent pair for all T-0 rows matching the run preset.",
        )
    for pair in rows:
        if pair == allowed:
            continue
        if pair in wrong_others:
            out.append(
                "Section 8 horizon blend: T-0 markdown row uses a blend digit pair from a different "
                "preset than this run; copy the injected fenced-table T-0 literal for the active preset.",
            )
        else:
            out.append(
                "Section 8 horizon blend: T-0 markdown row blend pair does not match the run preset "
                "or the standard preset catalog; fix the T-0 rows to the configured qual-then-quant literal.",
            )
    deduped: list[str] = []
    seen: set[str] = set()
    for item in out:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def horizon_blend_forbidden_literal_names() -> tuple[str, ...]:
    """Human-readable forbidden forms for synthesis output (documentation / UI)."""
    inv_t0 = "51" + ":" + "49"
    inv_t0_sp = "51" + " : " + "49"
    inv_pre = "45" + ":" + "55"
    inv_pre_sp = "45" + " : " + "55"
    return (
        inv_t0,
        inv_t0_sp,
        inv_pre,
        inv_pre_sp,
        "49 Quant : 51 Qual",
        "49 quant : 51 qual",
        "quant:qual",
        "quant : qual",
        "qualitative:quantitative",
        "qualitative : quantitative",
        "51% qualitative (paired with 49% quantitative for the T-0 / T+1..T+5 row)",
        "45% qualitative (paired with 55% quantitative for the T-3..T-1 row)",
    )


def horizon_blend_ratio_followups(
    synthesis_text: str,
    *,
    t0_blend_preset: T0BlendPreset = "default",
) -> list[str]:
    """Deterministic checks for section-8 horizon blend literals (preset-aware for T-0 table rows)."""
    out: list[str] = []
    text = synthesis_text
    has_49_51 = bool(_RE_BLEND_49_51.search(text))
    has_51_49 = bool(_RE_BLEND_51_49.search(text))
    has_55_45 = bool(_RE_BLEND_55_45.search(text))
    has_45_55 = bool(_RE_BLEND_45_55.search(text))

    if _RE_BAD_QUANT_FIRST_49_51.search(text):
        out.append(
            "Section 8 horizon blend: do not swap Quant/Qual lens labels against the table; "
            "use qual-then-quant wording with the fenced digit pair for T-0 / T+1..T+5.",
        )
    if _RE_BAD_QUANT_COLON_QUAL_LABEL.search(text):
        out.append(
            "Section 8 horizon blend: remove quant-then-qual label order; the blend column is "
            "always qual-then-quant (qualitative first, quantitative second).",
        )
    if _RE_BAD_QUAL_COLON_QUANT_WORDS.search(text):
        out.append(
            "Section 8 horizon blend: do not restate the blend as "
            "'qualitative-then-quantitative' with a colon; use the fenced markdown table "
            "or qual-then-quant with the canonical digit pairs only.",
        )
    if t0_blend_preset == "default" and _RE_BAD_PCT_ORDER_49_51_ROW.search(text):
        out.append(
            "Section 8 horizon blend: %-wording inverts qualitative vs quantitative shares "
            "for the T-0 / T+1..T+5 row; copy the fenced table (smaller share is qualitative).",
        )
    if _RE_BAD_PCT_ORDER_55_45_ROW.search(text) or _RE_BAD_PCT_ORDER_55_45_ROW_ALT.search(text):
        out.append(
            "Section 8 horizon blend: %-wording inverts qualitative vs quantitative shares "
            "for the T-3..T-1 row; copy the fenced table (larger share is qualitative).",
        )
    if has_49_51 and has_51_49:
        out.append(
            "Section 8 horizon blend: inconsistent blend literals for the T-0 / T+1..T+5 row "
            "(canonical digit pair plus its digit-inverted colon form); keep only the "
            "fenced-table pair with qualitative first.",
        )
    elif has_51_49:
        out.append(
            "Section 8 horizon blend: remove the digit-inverted colon pair for the T-0 / "
            "T+1..T+5 default; use only the fenced-table digits with qualitative first.",
        )
    if has_55_45 and has_45_55:
        out.append(
            "Section 8 horizon blend: inconsistent pre-event literals (canonical pair plus "
            "digit-inverted form for T-3..T-1); keep only the fenced-table pair.",
        )
    if has_45_55 and not has_55_45:
        out.append(
            "Section 8 horizon blend: remove the digit-inverted colon pair for T-3..T-1; "
            "use only the fenced-table digits with qualitative first.",
        )

    if t0_blend_preset == "quant_lean" and _RE_BLEND_60_40.search(text):
        out.append(
            "Section 8 horizon blend: remove digit-inverted 60 : 40; T-0 quant_lean preset uses "
            "40 : 60 (qual : quant).",
        )
    if t0_blend_preset == "quant_dominant" and _RE_BLEND_99_1.search(text):
        out.append(
            "Section 8 horizon blend: remove digit-inverted 99 : 1; T-0 quant_dominant preset uses "
            "1 : 99 (qual : quant).",
        )
    if t0_blend_preset == "qual_dominant" and _RE_BLEND_1_99.search(text):
        out.append(
            "Section 8 horizon blend: remove digit-inverted 1 : 99; T-0 qual_dominant preset uses "
            "99 : 1 (qual : quant).",
        )

    out.extend(_t0_preset_table_followups(text, t0_blend_preset=t0_blend_preset))

    deduped: list[str] = []
    seen: set[str] = set()
    for item in out:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


_GS_TILT = chr(0x03C3)
_PHI = chr(0x03A6)
_ENDASH = "\u2013"

# Human-readable list of phrases/patterns the synthesis validator rejects (see docs / tests).
QUALITATIVE_NUMERIC_TILT_FORBIDDEN_LITERALS_DOC: tuple[str, ...] = (
    "+5 point",
    "+5 pp",
    "+5 percentage point",
    f"+5{_ENDASH}<digit> (Unicode en dash U+2013, e.g. +5{_ENDASH}10)",
    "+5 to +10",
    "plus 5 (case-insensitive word boundary)",
    "+10 point",
    "+10 pp",
    "+10 percentage point",
    "mixed-quant tilt",
    "qualitative tilt",
    "point qualitative tilt",
    f"tilt within {_GS_TILT} bands",
    "tilt to scenario probabilities",
    "tilt to scenario weights",
)

_QUALITATIVE_NUMERIC_TILT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\+5\s*point", re.IGNORECASE), "+5 point"),
    (re.compile(r"\+5\s*pp\b", re.IGNORECASE), "+5 pp"),
    (re.compile(r"\+5\s*percentage\s*point", re.IGNORECASE), "+5 percentage point"),
    (re.compile(r"\+5\u2013\d"), "+5-en-dash+digit"),
    (re.compile(r"\+5\s*to\s*\+10", re.IGNORECASE), "+5 to +10"),
    (re.compile(r"(?i)\bplus\s+5\b"), "plus 5"),
    (re.compile(r"\+10\s*point", re.IGNORECASE), "+10 point"),
    (re.compile(r"\+10\s*pp\b", re.IGNORECASE), "+10 pp"),
    (re.compile(r"\+10\s*percentage\s*point", re.IGNORECASE), "+10 percentage point"),
    (re.compile(r"mixed-quant\s+tilt", re.IGNORECASE), "mixed-quant tilt"),
    (re.compile(r"qualitative\s+tilt", re.IGNORECASE), "qualitative tilt"),
    (re.compile(r"point\s+qualitative\s+tilt", re.IGNORECASE), "point qualitative tilt"),
    (
        re.compile(rf"tilt\s+within\s+({_GS_TILT}|sigma)\s+bands", re.IGNORECASE),
        f"tilt within {_GS_TILT} bands",
    ),
    (re.compile(r"tilt\s+to\s+scenario\s+probabilities", re.IGNORECASE), "tilt to scenario probabilities"),
    (re.compile(r"tilt\s+to\s+scenario\s+weights", re.IGNORECASE), "tilt to scenario weights"),
)


def qualitative_numeric_tilt_pattern_hits(text: str) -> list[str]:
    """Return labels for each forbidden qualitative-numeric-tilt pattern found in ``text``."""
    if not text:
        return []
    found: list[str] = []
    seen: set[str] = set()
    for rx, label in _QUALITATIVE_NUMERIC_TILT_PATTERNS:
        if rx.search(text) and label not in seen:
            seen.add(label)
            found.append(label)
    return found


def qualitative_numeric_tilt_followups(synthesis_text: str) -> list[str]:
    """Flag synthesis prose that reintroduces undefined +5/+10 qualitative numeric shifts."""
    hits = qualitative_numeric_tilt_pattern_hits(synthesis_text)
    if not hits:
        return []
    joined = "; ".join(hits)
    return [
        "Synthesis qualitative overlay: remove forbidden numeric qualitative-adjustment phrasing "
        f"({joined}). Use the canonical horizon blend ratio for narrative trust weighting only; "
        f"do not hand-shift probabilities, {_GS_TILT} band widths, or scenario-weight numerics—`prob_up_pct` "
        f"follows {_PHI} from bounded drift/{_GS_TILT} only. Rewrite or delete the offending sentence.",
    ]


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def iter_prompt_stack_text_files() -> tuple[Path, ...]:
    """Static prompt files and Python modules that embed system/user prompt text."""
    root = _repo_root()
    prompts_dir = root / "prompts"
    files: list[Path] = []
    for pat in ("*.md", "*.j2"):
        files.extend(sorted(prompts_dir.glob(pat)))
    for rel in (
        "equity_analyst/iterative.py",
        "equity_analyst/synthesizer.py",
        "equity_analyst/prompt_parts.py",
        "equity_analyst/provider_summarize.py",
    ):
        p = root / rel
        if p.is_file():
            files.append(p)
    return tuple(files)


def assert_prompt_stack_excludes_horizon_blend_inversions() -> None:
    """Regression: static prompts must not embed digit-inverted pairs or swapped labels."""
    needles: list[str] = [
        "51:49",
        "51 : 49",
        "45:55",
        "45 : 55",
        "49 Quant : 51 Qual",
        "49 quant : 51 qual",
    ]
    for path in iter_prompt_stack_text_files():
        raw = path.read_text(encoding="utf-8")
        lower = raw.lower()
        for n in needles:
            if n.lower() in lower:
                raise AssertionError(f"horizon blend inversion/leak in {path}: found {n!r}")
        if re.search(r"\bquant\s*:\s*qual\b", raw, re.IGNORECASE):
            raise AssertionError(f"horizon blend wrong lens order in {path}: quant:qual")
        if re.search(r"\bqualitative\s*:\s*quantitative\b", raw, re.IGNORECASE):
            raise AssertionError(
                f"horizon blend colon word-order inversion in {path}: qualitative:quantitative",
            )
