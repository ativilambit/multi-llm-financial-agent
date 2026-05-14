from __future__ import annotations

from pathlib import Path

from equity_analyst.prompt_parts import EQUITY_ANALYST_SYSTEM_PROMPT, _load_prompt_file
from equity_analyst.provider_summarize import summarize_system_prompt
from equity_analyst.synthesizer import SYNTHESIS_SYSTEM_PROMPT
from equity_analyst.synthesizer_blend import (
    assert_prompt_stack_excludes_horizon_blend_inversions,
    qualitative_numeric_tilt_pattern_hits,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS = REPO_ROOT / "prompts"


def _policy_invariants_text() -> str:
    return (PROMPTS / "policy" / "invariants.md").read_text(encoding="utf-8")


def _effective_equity_rules_surface() -> str:
    """Rules in `policy/invariants.md` plus the `.j2` body (matches render-time text surface)."""
    return _policy_invariants_text() + "\n" + (PROMPTS / "equity_analyst.j2").read_text(encoding="utf-8")


def _effective_synthesizer_system_prompt() -> str:
    return _load_prompt_file("synthesizer_system.md")


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
    raw = path.read_text(encoding="utf-8")
    assert raw.strip() != ""
    loaded = _load_prompt_file("synthesizer_system.md")
    assert loaded == SYNTHESIS_SYSTEM_PROMPT
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
    inv = _policy_invariants_text()
    synth = _effective_synthesizer_system_prompt()
    assert '{% include "policy/invariants.md" %}' in j2
    assert "| Horizon | Blend (qual : quant) | Notes |" in inv
    assert "| Horizon | Blend (qual : quant) | Notes |" in synth
    assert "__T0_BLEND_LITERAL__" in synth
    assert "{{ t0_blend_literal }}" in inv
    assert "55 : 45" in inv and "55 : 45" in synth
    assert inv.count("49 : 51") >= 1
    assert synth.count("49 : 51") >= 1
    t0_pre_j2 = (
        "| T-0 pre-open (event day, no intraday yet) | {{ t0_blend_literal }} | "
        "Mixed: options skew and pre-print positioning already price much of the setup; "
        "the default blend leans slightly quantitative for **trust weighting** while qualitative narrative still matters "
        "and the Pure-quant rule governs $/σ. |"
    )
    t0_pre_synth = (
        "| T-0 pre-open (event day, no intraday yet) | __T0_BLEND_LITERAL__ | "
        "Mixed: options skew and pre-print positioning already price much of the setup; "
        "the default blend leans slightly quantitative for **trust weighting** while qualitative narrative still matters "
        "and the Pure-quant rule governs $/σ. |"
    )
    t0_intra_j2 = (
        "| T-0 with same-day intraday available (mid-day / post-print / post-AMC) | {{ t0_blend_literal }} | "
        "After the tape and chain update, realized range and flow carry slightly more weight for **quantitative trust** "
        "in the narrative; qualitative drivers still shape story and scenarios; "
        "quantitative levels anchor exact $/σ math via the Pure-quant rule. |"
    )
    t0_intra_synth = (
        "| T-0 with same-day intraday available (mid-day / post-print / post-AMC) | __T0_BLEND_LITERAL__ | "
        "After the tape and chain update, realized range and flow carry slightly more weight for **quantitative trust** "
        "in the narrative; qualitative drivers still shape story and scenarios; "
        "quantitative levels anchor exact $/σ math via the Pure-quant rule. |"
    )
    t1_t5 = (
        "| T+1 to T+5 (after the event, with intraday history) | 49 : 51 | "
        "Realized post-event path and refreshed options data carry slightly more weight for **quantitative trust** "
        "in the narrative; qualitative drivers still inform scenario emphasis; "
        "exact $/σ bands remain quant-only. |"
    )
    assert t0_pre_j2 in inv and t0_pre_synth in synth
    assert t0_intra_j2 in inv and t0_intra_synth in synth
    assert t1_t5 in inv and t1_t5 in synth
    assert "40 : 60" not in inv and "40 : 60" not in synth
    assert "45 : 55" not in inv and "45 : 55" not in synth
    assert "Qualitative vs quantitative weighting" in inv
    assert "Qualitative vs quantitative weighting" in synth
    assert "default to the qualitative side" in inv and "unambiguous and recent" in j2
    assert "**default to the qualitative side**" in synth


def test_prompt_stack_has_no_horizon_blend_inversions() -> None:
    assert_prompt_stack_excludes_horizon_blend_inversions()


def test_prompt_md_j2_exclude_qualitative_numeric_tilt_patterns() -> None:
    """Regression: prompts must not reintroduce forbidden +5/+10 qualitative-tilt phrasing."""
    paths: list[Path] = []
    for path in sorted(PROMPTS.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in {".md", ".j2"}:
            continue
        paths.append(path)
    for path in paths:
        raw = path.read_text(encoding="utf-8")
        hits = qualitative_numeric_tilt_pattern_hits(raw)
        assert not hits, f"{path.relative_to(REPO_ROOT)} matched {hits!r}"


def test_section8_qualitative_evidence_subsections_and_limits() -> None:
    """Section 8 mandates sourced Qualitative evidence before short blend text."""
    j2 = (PROMPTS / "equity_analyst.j2").read_text(encoding="utf-8")
    assert "### Qualitative evidence" in j2
    assert "### Qualitative deep-dive & suggested blend (advisory)" in j2
    assert "### Horizon & blend application" in j2
    assert "### Directional resolution" in j2
    assert "minimum **6**" in j2 or "minimum 6" in j2
    assert "120 words" in j2
    assert "at most 4 sentences" in j2
    assert "Unable to verify from primary sources" in j2


def test_synthesizer_preserves_section8_qualitative_bullets() -> None:
    synth = (PROMPTS / "synthesizer_system.md").read_text(encoding="utf-8")
    assert "Preserve and dedupe" in synth and "Qualitative evidence" in synth
    assert "### Qualitative deep-dive & suggested blend (advisory)" in synth
    assert "methodology-only" in synth.lower()
    assert "conflicting" in synth.lower() and "sources" in synth.lower()


def test_qualitative_deep_dive_subsection_title_in_equity_j2_and_synthesizer_md() -> None:
    title = "### Qualitative deep-dive & suggested blend (advisory)"
    j2 = (PROMPTS / "equity_analyst.j2").read_text(encoding="utf-8")
    synth = (PROMPTS / "synthesizer_system.md").read_text(encoding="utf-8")
    assert title in j2
    assert title in synth
    assert j2.find("### Qualitative evidence") < j2.find(title) < j2.find("### Horizon & blend application")


def test_pure_quant_rule_in_equity_template_and_synthesizer() -> None:
    """Option pricing and sigma band widths are mandatory pure-quant; blend is narrative trust weighting only."""
    j2 = (PROMPTS / "equity_analyst.j2").read_text(encoding="utf-8")
    inv = _policy_invariants_text()
    synth = _effective_synthesizer_system_prompt()
    assert "Pure-quant rule" in j2
    assert "Pure-quant rule" in inv
    assert "Pure-quant rule" in synth
    assert "qualitative overlay does not move numbers" in inv.lower()
    assert "qualitative overlay does not move numbers" in synth.lower()
    assert "option pricing" in j2
    assert "option pricing" in inv
    assert "option pricing" in synth
    sigma = "\N{GREEK SMALL LETTER SIGMA}"
    assert f"{sigma} band widths" in j2
    assert f"{sigma} band widths" in inv
    assert f"{sigma} band widths" in synth


def test_unsourced_options_numbers_rule_in_equity_j2_and_synthesizer_md() -> None:
    inv = _policy_invariants_text()
    synth = _effective_synthesizer_system_prompt()
    assert "Unsourced numbers — options metrics (Pure-quant addendum)" in inv
    assert "options_chain_data" in inv
    assert "1-week-prior PCR" in inv
    assert "Unsourced numbers — options metrics (Pure-quant addendum)" in synth
    assert "historical chain data unavailable" in synth
    assert "strip it from the synthesis" in synth.lower()


def test_sigma_band_sanity_rules_in_equity_template_and_synthesizer() -> None:
    """Sigma band construction rules: variance-additive canonical, sqrt(t) fallback, checks."""
    surface = _effective_equity_rules_surface()
    synth = _effective_synthesizer_system_prompt()
    sig = chr(0x03C3)
    flat_surface = surface.replace("**", "")
    flat_synth = synth.replace("**", "")
    assert f"{sig}(T+N) = √(event_jump² + N · daily_vol²)" in flat_surface
    assert f"{sig}(T+N) = √(event_jump² + N · daily_vol²)" in flat_synth
    assert f"{sig}-scaling check (variance):" in surface
    assert "ej² + n·daily_vol²" in surface.replace("**", "")
    assert f"{sig}-scaling check (variance):" in synth
    assert "ej² + n·daily_vol²" in synth.replace("**", "")
    for needle in (
        "No fake same-day implied move",
        "Variance-additive event+diffusion decomposition",
        "√(target_DTE / chosen_expiry_DTE)",
        "HV30 √t scaling",
    ):
        assert needle in surface
        assert needle in synth


def test_equity_analyst_template_canonical_daily_vol_source_order() -> None:
    """Rule 2 must enumerate canonical daily_vol source order: HV30 -> realized -> calendar IV."""
    inv = _policy_invariants_text()
    assert "Canonical `daily_vol` source order" in inv
    hv30_idx = inv.find("**HV30**")
    realized_idx = inv.find("**Realized post-earnings daily vol**")
    calspread_idx = inv.find("**Forward IV calendar-spread**")
    assert 0 < hv30_idx < realized_idx < calspread_idx, (
        "Canonical daily_vol order must list HV30 first, realized post-earnings second, "
        "calendar-spread IV third"
    )
    minus = "\N{MINUS SIGN}"
    assert "annualized 30-day historical volatility / \N{SQUARE ROOT}252" in inv
    assert "the **last 4** earnings windows" in inv
    assert f"T_far {minus} T_event" in inv
    assert "State which source was used" in inv
    assert "daily_vol=3.15%/day (HV30 50.0% ann / \N{SQUARE ROOT}252)" in inv


def test_mandatory_sigma_literal_format_block_in_equity_j2_and_synthesizer_md() -> None:
    inv = _policy_invariants_text()
    synth = _effective_synthesizer_system_prompt()
    sg = chr(0x03C3)
    mandatory_equity = (
        "**MANDATORY (verifier will flag missing literals; you will be re-fanned-out to refine):** "
        f"Before showing any {sg} bands, output **exactly** these two lines in a fenced code block (any backticks), "
        "with the literal tokens `event_jump=` and `daily_vol=` in this exact form (no LaTeX, no Markdown italics, "
        "no Unicode multipliers):"
    )
    assert mandatory_equity in inv
    assert (
        "event_jump=<X.XX>% (<source description, e.g. May 15 weekly ATM straddle from options_chain_data>)"
        in inv
    )
    assert (
        "daily_vol=<Y.YY>%/day (<source: HV30 / realized post-earnings / IV-adjusted with multiplier>)" in inv
    )
    assert (
        "iv_crush_multiplier=<Z.ZZ> daily_vol_raw=<W.WW>%/day daily_vol=<Y.YY>%/day" in inv
    )
    assert mandatory_equity in synth
    assert (
        "MANDATORY machine-readable σ session table (downstream verifier)" in inv
        and "MANDATORY machine-readable σ session table (downstream verifier)" in synth
    )


def test_sigma_summary_json_contract_in_equity_and_synthesizer_prompts() -> None:
    inv = _policy_invariants_text()
    synth = _effective_synthesizer_system_prompt()
    assert "MANDATORY machine-readable" in inv and "sigma_summary" in inv
    assert "one_sigma_half_width_pct" in inv and "three_sigma_half_width_pct" in inv
    assert "anchor_type" in inv
    assert "MANDATORY machine-readable σ session table (downstream verifier)" in synth


def test_synthesizer_system_prompt_includes_per_provider_sigma_checks_paragraph() -> None:
    sigma = "\N{GREEK SMALL LETTER SIGMA}"
    synth = (PROMPTS / "synthesizer_system.md").read_text(encoding="utf-8")
    assert "per_provider_sigma_checks_markdown" in synth
    assert "passed=False" in synth
    assert f"Per-provider {sigma} variance pre-check" in synth
    assert "surface the disagreement explicitly" in synth.lower() or "explicitly" in synth


def test_equity_and_synthesizer_prompts_reference_server_computed_sigma_bands() -> None:
    j2 = (PROMPTS / "equity_analyst.j2").read_text(encoding="utf-8")
    synth = (PROMPTS / "synthesizer_system.md").read_text(encoding="utf-8")
    assert "computed_sigma_bands_available" in j2
    assert "computed_sigma_bands_markdown" in j2
    assert "verbatim" in j2.lower()
    assert "Pre-computed σ bands" in j2 or "pre-computed" in j2.lower()
    assert "Server-computed σ bands" in synth


def test_provider_summarize_system_prompt_file_exists_nonempty_and_matches_export() -> None:
    path = PROMPTS / "provider_summarize_system.md"
    assert path.is_file()
    raw = path.read_text(encoding="utf-8")
    assert raw.strip() != ""
    assert raw.rstrip() == summarize_system_prompt()


def test_synthesizer_prompt_has_mandatory_sigma_summary_block() -> None:
    synth = _effective_synthesizer_system_prompt()
    assert "Before showing any σ bands" in synth
    assert "MANDATORY machine-readable σ session table (downstream verifier)" in synth
    assert "sigma_summary" in synth
    assert "one_sigma_half_width_pct" in synth
    assert "Percent vs decimal" in synth


def test_section_9_requires_explicit_sigma_band_table_per_session() -> None:
    """Section 9 must repeat full 1σ/2σ/3σ per session before prediction prose."""
    j2 = (PROMPTS / "equity_analyst.j2").read_text(encoding="utf-8")
    assert "**σ-band table adjacency (mandatory):**" in j2
    assert "repeat the full 1σ / 2σ / 3σ band table from section 1 verbatim" in j2
    assert "three-line table immediately above" in j2
    assert "prose-only shorthand" in j2
    assert "*Prediction:*" in j2
    assert "**Wed May 13, 2026 — earnings day (BMO)**" in j2
    assert "- 3σ: $116.05 – $242.17 (±35.21%)" in j2


def test_section_11_includes_sigma_band_context_per_session() -> None:
    """Section 11 repeats full σ tables and pairs P(up) with 1σ on the same line."""
    j2 = (PROMPTS / "equity_analyst.j2").read_text(encoding="utf-8")
    assert "**σ bands adjacent to probabilities (mandatory):**" in j2
    assert "repeat the full 1σ / 2σ / 3σ band table from section 1 verbatim" in j2
    assert "**Wed May 13 (T0)** — 1σ: $158.09 – $200.13 (±11.74%) | P(up): 50.5%" in j2
    assert "not a substitute" in j2


def test_synthesizer_system_requires_explicit_bands_in_sections_9_11() -> None:
    synth = (PROMPTS / "synthesizer_system.md").read_text(encoding="utf-8")
    assert "**Sections 9 and 11 — σ band adjacency:**" in synth
    assert "never** need to scroll back to section 1" in synth
    assert "not** condensed prose-only references" in synth


def test_prompts_index_maps_every_prompt_file() -> None:
    index_path = PROMPTS / "INDEX.md"
    assert index_path.is_file()
    idx = index_path.read_text(encoding="utf-8")
    rels = sorted(
        p.relative_to(PROMPTS).as_posix()
        for p in PROMPTS.rglob("*")
        if p.is_file() and p.suffix in {".md", ".j2"}
    )
    assert rels, "expected at least one prompt file"
    for rel in rels:
        assert rel in idx, f"INDEX.md must mention {rel} (table row or inline path)"


def test_policy_invariants_nonempty() -> None:
    assert _policy_invariants_text().strip() != ""


def test_extracted_invariants_not_duplicated_on_disk() -> None:
    """Moved MUST blocks must not still appear in the old on-disk locations (single source)."""
    j2 = (PROMPTS / "equity_analyst.j2").read_text(encoding="utf-8")
    synth_src = (PROMPTS / "synthesizer_system.md").read_text(encoding="utf-8")
    inv = _policy_invariants_text()
    needle = (
        "**MANDATORY (verifier will flag missing literals; you will be re-fanned-out to refine):** "
        "Before showing any σ bands, output **exactly** these two lines in a fenced code block"
    )
    assert needle in inv
    assert needle not in j2
    assert needle not in synth_src
    assert "**MUST — literal horizon blend table:**" in inv
    assert "**MUST — literal horizon blend table:**" not in j2
    assert "**MUST — literal horizon blend table:**" not in synth_src
