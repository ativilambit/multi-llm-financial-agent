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


def test_provider_summarize_system_prompt_file_exists_nonempty_and_matches_export() -> None:
    path = PROMPTS / "provider_summarize_system.md"
    assert path.is_file()
    raw = path.read_text(encoding="utf-8")
    assert raw.strip() != ""
    assert raw.rstrip() == summarize_system_prompt()
