from __future__ import annotations

from equity_analyst.iterative import augment_verifier_result_with_sigma_structural_checks
from equity_analyst.synthesizer_blend import (
    QUALITATIVE_NUMERIC_TILT_FORBIDDEN_LITERALS_DOC,
    horizon_blend_ratio_followups,
    qualitative_numeric_tilt_followups,
    qualitative_numeric_tilt_pattern_hits,
    suggested_blend_consistency_followups,
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


def test_suggested_blend_consistency_followup_when_missing_final_table() -> None:
    bundle = (
        "## p1\n"
        "### Qualitative deep-dive & suggested blend (advisory)\n"
        "| Horizon bucket | Suggested qual : quant (two ints summing to 100) | Notes |\n"
        "| T+1 to T+5 | 52 : 48 | a |\n\n"
        "### Horizon & blend application\n"
        "## p2\n"
        "### Qualitative deep-dive & suggested blend (advisory)\n"
        "| Horizon bucket | Suggested qual : quant (two ints summing to 100) | Notes |\n"
        "| T+1 to T+5 | 40 : 60 | b |\n\n"
        "### Horizon & blend application\n"
    )
    syn = "Synthesis without the requested heading."
    msgs = suggested_blend_consistency_followups(syn, provider_iteration_bundle=bundle)
    assert msgs and "Final suggested blend" in msgs[0]


def test_suggested_blend_consistency_followup_suppressed_when_heading_present() -> None:
    bundle = (
        "## p1\n"
        "### Qualitative deep-dive & suggested blend (advisory)\n"
        "| Horizon bucket | Suggested qual : quant (two ints summing to 100) | Notes |\n"
        "| T+1 to T+5 | 52 : 48 | a |\n\n"
        "### Horizon & blend application\n"
        "## p2\n"
        "### Qualitative deep-dive & suggested blend (advisory)\n"
        "| Horizon bucket | Suggested qual : quant (two ints summing to 100) | Notes |\n"
        "| T+1 to T+5 | 40 : 60 | b |\n\n"
        "### Horizon & blend application\n"
    )
    syn = "### Final suggested blend (advisory — consensus)\n\n| Horizon bucket | ... |\n"
    assert suggested_blend_consistency_followups(syn, provider_iteration_bundle=bundle) == []


def test_augment_verifier_includes_suggested_blend_soft_followup() -> None:
    bundle = (
        "## p1\n"
        "### Qualitative deep-dive & suggested blend (advisory)\n"
        "| Horizon bucket | Suggested qual : quant (two ints summing to 100) | Notes |\n"
        "| T+1 to T+5 | 52 : 48 | a |\n\n"
        "### Horizon & blend application\n"
        "## p2\n"
        "### Qualitative deep-dive & suggested blend (advisory)\n"
        "| Horizon bucket | Suggested qual : quant (two ints summing to 100) | Notes |\n"
        "| T+1 to T+5 | 40 : 60 | b |\n\n"
        "### Horizon & blend application\n"
    )
    syn = "Synthesis body only.\n### Horizon & blend application\n49 : 51\n"
    base = {"verified": [], "contradicted": [], "unverifiable": []}
    out = augment_verifier_result_with_sigma_structural_checks(
        syn,
        base,
        provider_iteration_bundle=bundle,
    )
    joined = " ".join(out.get("unverifiable") or [])
    assert "Final suggested blend" in joined


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


def test_qualitative_numeric_tilt_enforcement_disabled() -> None:
    """Qualitative-numeric tilt patterns are no longer flagged (stable API; doc tuple empty)."""
    assert QUALITATIVE_NUMERIC_TILT_FORBIDDEN_LITERALS_DOC == ()
    assert qualitative_numeric_tilt_pattern_hits("+5 pp fudge tilt narrative") == []
    assert qualitative_numeric_tilt_followups("+5 pp fudge tilt narrative") == []


def test_qualitative_numeric_tilt_followups_passes_clean_blend_prose() -> None:
    text = (
        "### Horizon & blend application\n"
        "We use qual:quant = 49 : 51 for T+1..T+5 and narrate PEAD vs skew without numeric fudge factors.\n"
    )
    assert qualitative_numeric_tilt_followups(text) == []
    assert qualitative_numeric_tilt_pattern_hits(text) == []
