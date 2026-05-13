from __future__ import annotations

import re

# Horizon blend rows are always stated as **qual : quant** (qualitative first).
_RE_BLEND_49_51 = re.compile(r"\b49\s*:\s*51\b")
_RE_BLEND_51_49 = re.compile(r"\b51\s*:\s*49\b")
_RE_BLEND_55_45 = re.compile(r"\b55\s*:\s*45\b")
_RE_BLEND_45_55 = re.compile(r"\b45\s*:\s*55\b")
# Common LLM mistake: swaps lens names vs the table while keeping the same digits.
_RE_BAD_QUANT_FIRST_49_51 = re.compile(r"49\s+Quant\s*:\s*51\s+Qual", re.IGNORECASE)


def horizon_blend_ratio_followups(synthesis_text: str) -> list[str]:
    """Deterministic checks for section-8 horizon blend literals.

    The equity + synthesizer prompts define default weights as **qual : quant**
    (qualitative first, quantitative second). For post-default horizons the
    digits are **49 : 51**; writing **51 : 49** inverts the pair relative to
    that convention and must be flagged for refinement.
    """
    out: list[str] = []
    text = synthesis_text
    has_49_51 = bool(_RE_BLEND_49_51.search(text))
    has_51_49 = bool(_RE_BLEND_51_49.search(text))
    has_55_45 = bool(_RE_BLEND_55_45.search(text))
    has_45_55 = bool(_RE_BLEND_45_55.search(text))

    if _RE_BAD_QUANT_FIRST_49_51.search(text):
        out.append(
            "Section 8 horizon blend: do not write '49 Quant : 51 Qual'; canonical wording is "
            "qual:quant = 49:51 (qualitative first, quantitative second).",
        )
    if has_49_51 and has_51_49:
        out.append(
            "Section 8 horizon blend: inconsistent blend literals (both 49:51 and 51:49); "
            "use exactly qual:quant = 49:51 everywhere for that row (51:49 is digit-inverted).",
        )
    elif has_51_49:
        out.append(
            "Section 8 horizon blend: remove literal 51:49; canonical T-0 / T+1..T+5 row is "
            "qual:quant = 49:51 only (do not invert the digits).",
        )
    if has_55_45 and has_45_55:
        out.append(
            "Section 8 horizon blend: inconsistent pre-event literals (both 55:45 and 45:55); "
            "use exactly qual:quant = 55:45 for T-3..T-1.",
        )
    if has_45_55 and not has_55_45:
        out.append(
            "Section 8 horizon blend: remove literal 45:55 for the T-3..T-1 row; canonical is "
            "qual:quant = 55:45 (do not invert the digits).",
        )

    deduped: list[str] = []
    seen: set[str] = set()
    for item in out:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped
