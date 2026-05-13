from __future__ import annotations

from equity_analyst.iterative import augment_verifier_result_with_sigma_structural_checks
from equity_analyst.synthesizer_blend import horizon_blend_ratio_followups


def test_horizon_blend_followups_flags_both_49_51_and_51_49() -> None:
    text = (
        "### Horizon & blend application\n"
        "We use qual:quant = 49:51 for T+1..T+5. Elsewhere we wrote 51:49 by mistake.\n"
    )
    msgs = horizon_blend_ratio_followups(text)
    assert msgs, "expected at least one follow-up for inconsistent literals"
    joined = " ".join(msgs).lower()
    assert "49:51" in joined and "51:49" in joined


def test_horizon_blend_followups_passes_canonical_only() -> None:
    text = (
        "### Horizon & blend application\n"
        "Given same-day intraday, we apply qual:quant = 49 : 51 for T0 through T+5.\n"
    )
    assert horizon_blend_ratio_followups(text) == []


def test_horizon_blend_followups_flags_quant_first_label_swap() -> None:
    text = "We use the 49 Quant : 51 Qual blend for T+2."
    msgs = horizon_blend_ratio_followups(text)
    assert msgs and "49 quant" in msgs[0].lower()


def test_horizon_blend_followups_flags_51_49_alone() -> None:
    assert horizon_blend_ratio_followups("Blend is 51:49 for T+1.") != []


def test_augment_verifier_includes_blend_messages() -> None:
    syn = (
        "Synthesis body\n"
        "### Horizon & blend application\n"
        "Row one 49:51 and row two 51:49.\n"
    )
    base = {"verified": [], "contradicted": [], "unverifiable": []}
    out = augment_verifier_result_with_sigma_structural_checks(syn, base)
    unv = " ".join(out.get("unverifiable") or []).lower()
    assert "blend" in unv and "49:51" in unv and "51:49" in unv
