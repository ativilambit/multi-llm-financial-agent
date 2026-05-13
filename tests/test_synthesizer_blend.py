from __future__ import annotations

import pytest

from equity_analyst.iterative import augment_verifier_result_with_sigma_structural_checks
from equity_analyst.synthesizer_blend import (
    QUALITATIVE_NUMERIC_TILT_FORBIDDEN_LITERALS_DOC,
    horizon_blend_ratio_followups,
    qualitative_numeric_tilt_followups,
    qualitative_numeric_tilt_pattern_hits,
)


def test_horizon_blend_followups_flags_both_49_51_and_51_49() -> None:
    text = (
        "### Horizon & blend application\n"
        "We use qual:quant = 49:51 for T+1..T+5. Elsewhere we wrote 51:49 by mistake.\n"
    )
    msgs = horizon_blend_ratio_followups(text)
    assert msgs, "expected at least one follow-up for inconsistent literals"
    joined = " ".join(msgs).lower()
    assert "blend" in joined and "inconsistent" in joined


def test_horizon_blend_followups_passes_canonical_only() -> None:
    text = (
        "### Horizon & blend application\n"
        "Given same-day intraday, we apply qual:quant = 49 : 51 for T0 through T+5.\n"
    )
    assert horizon_blend_ratio_followups(text) == []


def test_horizon_blend_followups_flags_quant_first_label_swap() -> None:
    text = "We use the 49 Quant : 51 Qual blend for T+2."
    msgs = horizon_blend_ratio_followups(text)
    assert msgs and "lens" in msgs[0].lower()


def test_horizon_blend_followups_flags_51_49_alone() -> None:
    assert horizon_blend_ratio_followups("Blend is 51:49 for T+1.") != []


def test_horizon_blend_followups_flags_quant_colon_qual_label() -> None:
    text = "Section 8 applies quant:qual = 49 : 51 for T+1."
    msgs = horizon_blend_ratio_followups(text)
    assert msgs and "quant-then-qual" in " ".join(msgs).lower()


def test_horizon_blend_followups_flags_qualitative_colon_quantitative() -> None:
    text = "Default blend is qualitative:quantitative = 49 : 51 for T+2."
    msgs = horizon_blend_ratio_followups(text)
    assert msgs and "qualitative-then-quantitative" in " ".join(msgs).lower()


def test_horizon_blend_followups_flags_inverted_pct_49_row() -> None:
    text = "Horizon row implies 51% qualitative vs 49% quantitative for T+2."
    msgs = horizon_blend_ratio_followups(text)
    assert msgs and "%" in msgs[0]


def test_horizon_blend_followups_flags_inverted_pct_55_row() -> None:
    text = "Pre-event we use 45% qualitative and 55% quantitative for T-2."
    msgs = horizon_blend_ratio_followups(text)
    assert msgs


def test_augment_verifier_includes_blend_messages() -> None:
    syn = (
        "Synthesis body\n"
        "### Horizon & blend application\n"
        "Row one 49:51 and row two 51:49.\n"
    )
    base = {"verified": [], "contradicted": [], "unverifiable": []}
    out = augment_verifier_result_with_sigma_structural_checks(syn, base)
    unv = " ".join(out.get("unverifiable") or []).lower()
    assert "blend" in unv and "inconsistent" in unv


_SIG = chr(0x03C3)


@pytest.mark.parametrize(
    "snippet",
    [
        "Applied +5 point narrative fudge to P(up).",
        "Apply +5 pp to the scenario mix.",
        "+5 percentage point bump",
        "+5 to +10 versus baseline",
        "Plus 5 if you trust qual more",
        "+10 point adjustment",
        "+10 pp bump",
        "+10 percentage point shift",
        "mixed-quant tilt language",
        "qualitative tilt to weights",
        "point qualitative tilt mechanics",
        f"tilt within {_SIG} bands narrative",
        "tilt within sigma bands narrative",
        "tilt to scenario probabilities",
        "tilt to scenario weights",
    ],
)
def test_qualitative_numeric_tilt_followups_detects_forbidden_snippets(snippet: str) -> None:
    msgs = qualitative_numeric_tilt_followups(snippet)
    assert msgs
    assert "forbidden" in msgs[0].lower() or "remove" in msgs[0].lower()


def test_qualitative_numeric_tilt_doc_tuple_covers_pattern_count() -> None:
    """Keep documentation tuple roughly aligned with implemented pattern labels."""
    assert len(QUALITATIVE_NUMERIC_TILT_FORBIDDEN_LITERALS_DOC) >= 10


def test_qualitative_numeric_tilt_followups_en_dash_range() -> None:
    text = "We add +5–10 points to qualitative trust."
    assert qualitative_numeric_tilt_followups(text)


def test_qualitative_numeric_tilt_followups_passes_clean_blend_prose() -> None:
    text = (
        "### Horizon & blend application\n"
        "We use qual:quant = 49 : 51 for T+1..T+5 and narrate PEAD vs skew without numeric fudge factors.\n"
    )
    assert qualitative_numeric_tilt_followups(text) == []
    assert qualitative_numeric_tilt_pattern_hits(text) == []


def test_augment_verifier_includes_qualitative_numeric_tilt_messages() -> None:
    syn = "Consensus:\nWe tilt qualitative by +5 to +10 percentage points vs the row.\n"
    base = {"verified": [], "contradicted": [], "unverifiable": []}
    out = augment_verifier_result_with_sigma_structural_checks(syn, base)
    unv = " ".join(out.get("unverifiable") or []).lower()
    assert "qualitative overlay" in unv
    assert "forbidden" in unv or "numeric" in unv
