from __future__ import annotations

from pathlib import Path

from equity_analyst.prompt_parts import EQUITY_ANALYST_SYSTEM_PROMPT
from equity_analyst.synthesizer import SYNTHESIS_SYSTEM_PROMPT

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS = REPO_ROOT / "prompts"


def test_equity_analyst_system_prompt_file_exists_nonempty_and_matches_export() -> None:
    path = PROMPTS / "equity_analyst_system.md"
    assert path.is_file()
    raw = path.read_text(encoding="utf-8")
    assert raw.strip() != ""
    assert raw.rstrip() == EQUITY_ANALYST_SYSTEM_PROMPT


def test_synthesizer_system_prompt_file_exists_nonempty_and_matches_export() -> None:
    path = PROMPTS / "synthesizer_system.md"
    assert path.is_file()
    raw = path.read_text(encoding="utf-8")
    assert raw.strip() != ""
    assert raw.rstrip() == SYNTHESIS_SYSTEM_PROMPT
