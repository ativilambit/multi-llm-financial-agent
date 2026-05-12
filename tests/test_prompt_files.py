from __future__ import annotations

from pathlib import Path

from equity_analyst.prompt_parts import EQUITY_ANALYST_SYSTEM_PROMPT
from equity_analyst.provider_summarize import summarize_system_prompt
from equity_analyst.synthesizer import SYNTHESIS_SYSTEM_PROMPT

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS = REPO_ROOT / "prompts"


def test_equity_analyst_system_prompt_file_exists_nonempty_and_matches_export() -> None:
    path = PROMPTS / "equity_analyst_system.md"
    assert path.is_file()
    raw = path.read_text(encoding="utf-8")
    assert raw.strip() != ""
    assert raw.rstrip() == EQUITY_ANALYST_SYSTEM_PROMPT
    # Heuristic token estimate (len/4): keep persona long enough for provider prompt-caching minima.
    assert len(EQUITY_ANALYST_SYSTEM_PROMPT) // 4 >= 1400


def test_synthesizer_system_prompt_file_exists_nonempty_and_matches_export() -> None:
    path = PROMPTS / "synthesizer_system.md"
    assert path.is_file()
    raw = path.read_text(encoding="utf-8")
    assert raw.strip() != ""
    assert raw.rstrip() == SYNTHESIS_SYSTEM_PROMPT
    assert "preserve ALL standard deviation levels" in SYNTHESIS_SYSTEM_PROMPT
    sigma = "\N{GREEK SMALL LETTER SIGMA}"
    assert f"1{sigma}" in SYNTHESIS_SYSTEM_PROMPT
    assert f"2{sigma}" in SYNTHESIS_SYSTEM_PROMPT
    assert f"3{sigma}" in SYNTHESIS_SYSTEM_PROMPT


def test_synthesizer_system_prompt_covers_same_day_sd_anchor() -> None:
    raw = (PROMPTS / "synthesizer_system.md").read_text(encoding="utf-8")
    assert "same_day_intraday_available" in raw
    assert "intraday_min" in raw


def test_qualitative_weighting_in_equity_analyst_template_and_synthesizer() -> None:
    """Prompts include horizon-aware qualitative vs quantitative blend table."""
    j2 = (PROMPTS / "equity_analyst.j2").read_text(encoding="utf-8")
    synth = (PROMPTS / "synthesizer_system.md").read_text(encoding="utf-8")
    assert "| Horizon | Default blend (qual : quant) | Rationale |" in j2
    assert "| Horizon | Default blend (qual : quant) | Rationale |" in synth
    for ratio in ("55 : 45", "51 : 49", "40 : 60", "45 : 55"):
        assert ratio in j2
        assert ratio in synth
    assert "Qualitative vs quantitative weighting" in j2
    assert "Qualitative vs quantitative weighting" in synth
    assert "default to the qualitative side" in j2 and "unambiguous and recent" in j2
    assert "**default to the qualitative side**" in synth


def test_section8_qualitative_evidence_subsections_and_limits() -> None:
    """Section 8 mandates sourced Qualitative evidence before short blend text."""
    j2 = (PROMPTS / "equity_analyst.j2").read_text(encoding="utf-8")
    assert "### Qualitative evidence" in j2
    assert "### Horizon & blend application" in j2
    assert "### Directional resolution" in j2
    assert "minimum **6**" in j2 or "minimum 6" in j2
    assert "120 words" in j2
    assert "at most 4 sentences" in j2
    assert "Unable to verify from primary sources" in j2


def test_synthesizer_preserves_section8_qualitative_bullets() -> None:
    synth = (PROMPTS / "synthesizer_system.md").read_text(encoding="utf-8")
    assert "Preserve and dedupe" in synth and "Qualitative evidence" in synth
    assert "methodology-only" in synth.lower()
    assert "conflicting" in synth.lower() and "sources" in synth.lower()


def test_pure_quant_rule_in_equity_template_and_synthesizer() -> None:
    """Option pricing and sigma band widths are mandatory pure-quant; blend is for tilt/weights."""
    j2 = (PROMPTS / "equity_analyst.j2").read_text(encoding="utf-8")
    synth = (PROMPTS / "synthesizer_system.md").read_text(encoding="utf-8")
    assert "Pure-quant rule" in j2
    assert "Pure-quant rule" in synth
    assert "option pricing" in j2
    assert "option pricing" in synth
    sigma = "\N{GREEK SMALL LETTER SIGMA}"
    assert f"{sigma} band widths" in j2
    assert f"{sigma} band widths" in synth


def test_sigma_band_sanity_rules_in_equity_template_and_synthesizer() -> None:
    """Sigma band construction rules: variance-additive canonical, sqrt(t) fallback, checks."""
    j2 = (PROMPTS / "equity_analyst.j2").read_text(encoding="utf-8")
    synth = (PROMPTS / "synthesizer_system.md").read_text(encoding="utf-8")
    sig = chr(0x03C3)
    flat_j2 = j2.replace("**", "")
    flat_synth = synth.replace("**", "")
    assert f"{sig}(T+N) = √(event_jump² + N · daily_vol²)" in flat_j2
    assert f"{sig}(T+N) = √(event_jump² + N · daily_vol²)" in flat_synth
    assert f"{sig}-scaling check (variance):" in j2
    assert f"{sig}²(T+N)" in j2 and f"{sig}²(T+1)" in j2
    assert "daily_vol² = Y.YY" in j2
    assert f"{sig}-scaling check (variance):" in synth
    assert f"{sig}²(T+N)" in synth and f"{sig}²(T+1)" in synth
    assert "daily_vol² = Y.YY" in synth
    for needle in (
        "No fake same-day implied move",
        "Variance-additive event+diffusion decomposition",
        "√(target_DTE / chosen_expiry_DTE)",
        "HV30 √t scaling",
    ):
        assert needle in j2
        assert needle in synth


def test_provider_summarize_system_prompt_file_exists_nonempty_and_matches_export() -> None:
    path = PROMPTS / "provider_summarize_system.md"
    assert path.is_file()
    raw = path.read_text(encoding="utf-8")
    assert raw.strip() != ""
    assert raw.rstrip() == summarize_system_prompt()
