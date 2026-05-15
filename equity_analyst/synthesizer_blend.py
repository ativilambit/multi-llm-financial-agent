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


# Empty: qualitative-numeric tilt pattern enforcement removed (names retained for API stability).
QUALITATIVE_NUMERIC_TILT_FORBIDDEN_LITERALS_DOC: tuple[str, ...] = ()


def qualitative_numeric_tilt_pattern_hits(_text: str) -> list[str]:
    """Return labels for forbidden qualitative-numeric-tilt patterns (none are enforced)."""
    return []


SECTION_8B_HEAD = "### Qualitative deep-dive & suggested blend (advisory)"
SECTION_8C_HEAD = "### Horizon & blend application"
FINAL_ADVISORY_BLEND_HEAD_NEEDLE = "final suggested blend (advisory"

# Verifier prompt / deterministic nudge: flag wide cross-provider advisory spreads without consensus heading.
ADVISORY_BLEND_PROVIDER_SPREAD_THRESHOLD_POINTS = 10


def _slice_section8b(text: str) -> str:
    i = text.find(SECTION_8B_HEAD)
    if i < 0:
        return ""
    rest = text[i:]
    j = rest.find(SECTION_8C_HEAD)
    if j < 0:
        return rest
    return rest[:j]


def _bucket_id_for_horizon_cell(cell: str) -> str | None:
    c = cell.lower().replace("\u2212", "-").replace("\u2013", "-")
    c_ns = c.replace(" ", "")
    if "t+1" in c_ns:
        return "tplus"
    if "pre-open" in c or "pre open" in c:
        return "preopen"
    if "same-dayintraday" in c_ns or "samedayintraday" in c_ns:
        return "intra"
    if ("intraday" in c or "post-print" in c or "post-amc" in c) and (
        "t-0" in c_ns or "t0" in c_ns or "event day" in c
    ):
        return "intra"
    if "t-3" in c or "days before" in c:
        return "premkt"
    return None


def _table_row_bucket_and_pair(line: str) -> tuple[str, tuple[int, int]] | None:
    cells = _markdown_row_cells(line)
    if not cells or len(cells) < 2:
        return None
    bid = _bucket_id_for_horizon_cell(cells[0])
    if bid is None:
        return None
    pair = _parse_qual_quant_pair(cells[1])
    if pair is None:
        return None
    q, u = pair
    if not (0 <= q <= 100 and 0 <= u <= 100 and q + u == 100):
        return None
    return bid, (q, u)


def _fallback_prose_bucket_pairs(slice_text: str) -> dict[str, tuple[int, int]]:
    """Parse ``qual : quant`` from non-table lines in section 8B (bullet-style advisories)."""
    out: dict[str, tuple[int, int]] = {}
    for line in slice_text.splitlines():
        low = line.lower().replace("\u2212", "-").replace("\u2013", "-")
        m = re.search(r"\b(\d{1,3})\s*:\s*(\d{1,3})\b", line)
        if not m:
            continue
        q, u = int(m.group(1)), int(m.group(2))
        if not (0 <= q <= 100 and 0 <= u <= 100 and q + u == 100):
            continue
        bid: str | None = None
        low_ns = low.replace(" ", "")
        if "t+1" in low_ns:
            bid = "tplus"
        elif "pre-open" in low or "pre open" in low:
            bid = "preopen"
        elif "intraday" in low or "post-print" in low or "post-amc" in low:
            bid = "intra"
        elif "t-3" in low or "daysbefore" in low_ns or "beforeevent" in low_ns:
            bid = "premkt"
        if bid is not None:
            out[bid] = (q, u)
    return out


def _pairs_from_section8b_slice(slice_text: str) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for line in slice_text.splitlines():
        got = _table_row_bucket_and_pair(line)
        if got is None:
            continue
        bid, pair = got
        out[bid] = pair
    for bid, pair in _fallback_prose_bucket_pairs(slice_text).items():
        out.setdefault(bid, pair)
    return out


def _iter_provider_chunks(bundle: str) -> list[str]:
    t = bundle.strip()
    if not t:
        return []
    parts = re.split(r"(?m)^##\s+\S+\s*\n", t)
    if len(parts) <= 1:
        return [t]
    chunks: list[str] = []
    if parts[0].strip():
        chunks.append(parts[0])
    chunks.extend(p for p in parts[1:] if p.strip())
    return chunks or [t]


def max_advisory_qual_spread_across_providers(bundle: str) -> int:
    """Largest |Δqual| for a single horizon bucket across provider section-8B table rows (0 if unknown)."""
    per_bucket: dict[str, list[int]] = {}
    for chunk in _iter_provider_chunks(bundle):
        s8 = _slice_section8b(chunk)
        if not s8:
            continue
        rowmap = _pairs_from_section8b_slice(s8)
        for bid, (q, _u) in rowmap.items():
            per_bucket.setdefault(bid, []).append(q)
    best = 0
    for quals in per_bucket.values():
        if len(quals) < 2:
            continue
        span = max(quals) - min(quals)
        if span > best:
            best = span
    return best


def distinct_sum100_pairs_in_section8b(bundle: str) -> set[tuple[int, int]]:
    """Distinct qual:quant pairs (summing to 100) parsed from section-8B markdown table rows."""
    out: set[tuple[int, int]] = set()
    for chunk in _iter_provider_chunks(bundle):
        s8 = _slice_section8b(chunk)
        for _bid, pair in _pairs_from_section8b_slice(s8).items():
            out.add(pair)
    return out


def suggested_blend_consistency_followups(
    synthesis_text: str,
    *,
    provider_iteration_bundle: str = "",
) -> list[str]:
    """Low-severity nudge when providers disagree on advisory blends but synthesis omits the final consensus heading."""
    if not provider_iteration_bundle.strip():
        return []
    if FINAL_ADVISORY_BLEND_HEAD_NEEDLE in synthesis_text.lower():
        return []
    pairs = distinct_sum100_pairs_in_section8b(provider_iteration_bundle)
    spread = max_advisory_qual_spread_across_providers(provider_iteration_bundle)
    if len(pairs) < 2 and spread <= ADVISORY_BLEND_PROVIDER_SPREAD_THRESHOLD_POINTS:
        return []
    return [
        "(follow-up) Section 8 advisory blends: providers diverge on suggested qual:quant table rows; "
        "add `### Final suggested blend (advisory — consensus)` with a four-row reconciled grid per synthesizer merge rules, "
        "and per-row Dissent notes listing each dissenting provider with that provider's stated qual:quant pair when available.",
    ]


def qualitative_numeric_tilt_followups(_synthesis_text: str) -> list[str]:
    """Reserved hook for qualitative-numeric tilt checks (currently returns no follow-ups)."""
    return []


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def iter_prompt_stack_text_files() -> tuple[Path, ...]:
    """Static prompt files and Python modules that embed system/user prompt text."""
    root = _repo_root()
    prompts_dir = root / "prompts"
    files: list[Path] = []
    for pat in ("*.md", "*.j2"):
        files.extend(sorted(prompts_dir.glob(pat)))
    inv = prompts_dir / "policy" / "invariants.md"
    if inv.is_file():
        files.append(inv)
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
