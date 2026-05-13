from __future__ import annotations

from pathlib import Path

import pytest

from equity_analyst.config import RunConfig
from equity_analyst.iterative import augment_verifier_result_with_sigma_structural_checks
from equity_analyst.prompt_parts import _load_prompt_file
from equity_analyst.prompting import render_prompt
from equity_analyst.synthesizer_blend import (
    T0_BLEND_QUAL_QUANT,
    format_t0_blend_qual_quant_literal,
    inject_t0_blend_into_synthesizer_system_prompt,
    normalize_t0_blend_preset,
)


def _minimal_cfg_dict(**extra: object) -> dict[str, object]:
    return {
        "symbol": "X",
        "today_date": "d",
        "today_session": "s",
        "earnings_date": "e",
        "target_dates": [],
        "next_trading_day": "n",
        "followup_open_date": "f",
        "providers": ["openai"],
        **extra,
    }


@pytest.mark.parametrize(
    "preset,literal,forbidden_substrings",
    [
        ("default", "49 : 51", ("40 : 60", "1 : 99", "99 : 1")),
        ("quant_lean", "40 : 60", ("1 : 99", "99 : 1")),
        ("quant_dominant", "1 : 99", ("40 : 60", "99 : 1")),
        ("qual_dominant", "99 : 1", ("40 : 60", "1 : 99")),
    ],
)
def test_inject_synthesizer_prompt_substitutes_t0_literal(
    preset: str,
    literal: str,
    forbidden_substrings: tuple[str, ...],
) -> None:
    raw = _load_prompt_file("synthesizer_system.md")
    assert "__T0_BLEND_LITERAL__" in raw
    out = inject_t0_blend_into_synthesizer_system_prompt(raw, normalize_t0_blend_preset(preset))
    assert literal in out
    assert "__T0_BLEND_LITERAL__" not in out
    for sub in forbidden_substrings:
        assert sub not in out
    assert "49 : 51" in out


@pytest.mark.parametrize("preset", ["default", "quant_lean", "quant_dominant", "qual_dominant"])
def test_rendered_equity_prompt_contains_only_active_t0_literal(preset: str) -> None:
    cfg = RunConfig.model_validate({**_minimal_cfg_dict(), "t0_blend_preset": preset})
    rp = render_prompt(cfg, Path("prompts/equity_analyst.j2"))
    lit = format_t0_blend_qual_quant_literal(normalize_t0_blend_preset(preset))
    assert lit in rp.user_message_text
    for k, pair in T0_BLEND_QUAL_QUANT.items():
        if k == preset:
            continue
        other = f"{pair[0]} : {pair[1]}"
        if other == "49 : 51":
            assert "49 : 51" in rp.user_message_text
            continue
        assert other not in rp.user_message_text


def test_augment_verifier_flags_wrong_t0_table_for_quant_dominant() -> None:
    syn = (
        "### Horizon & blend application\n"
        "| Horizon | Blend (qual : quant) | Notes |\n"
        "|---|---|---|\n"
        "| T-0 pre-open (event day, no intraday yet) | 49 : 51 | x |\n"
    )
    base = {"verified": [], "contradicted": [], "unverifiable": []}
    out = augment_verifier_result_with_sigma_structural_checks(
        syn,
        base,
        t0_blend_preset="quant_dominant",
    )
    joined = " ".join(out.get("unverifiable") or []).lower()
    assert "t-0" in joined or "preset" in joined


def test_normalize_t0_blend_preset_invalid() -> None:
    with pytest.raises(ValueError):
        normalize_t0_blend_preset("nope")
