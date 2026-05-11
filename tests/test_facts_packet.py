from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from equity_analyst.config import RunConfig
from equity_analyst.facts_packet import (
    _facts_packet_fallback_markdown,
    extract_facts_packet,
    facts_frozen_user_prefix,
    write_facts_packet,
)
from equity_analyst.providers.registry import ProviderRegistry
from equity_analyst.types import ProviderResponse, ProviderUsage

_SIGMA = "\u03c3"
_FACTS_EXTRACT_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "facts_extract_system.md"


def test_facts_frozen_user_prefix_format() -> None:
    md = "# Market facts (frozen from iteration 1)\n\n- Last verified close: $10.00\n"
    p = facts_frozen_user_prefix(facts_markdown=md)
    assert "FACTS (frozen from iteration 1" in p
    assert "do NOT re-fetch via web_search" in p
    assert "# TASK" in p
    assert "$10.00" in p


def test_write_facts_packet_persists(tmp_path: Path) -> None:
    content = "# Market facts (frozen from iteration 1)\n\n- Row\n"
    path = write_facts_packet(tmp_path, content)
    assert path.name == "facts_packet.md"
    assert path.read_text(encoding="utf-8") == content


class _StubGemini:
    name = "gemini"

    def __init__(self, out_text: str) -> None:
        self._out = out_text

    async def generate(
        self,
        prompt: str,
        *,
        enable_web_search: bool = True,
        max_output_tokens: int | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        assert "MNDY" in prompt
        assert enable_web_search is False
        return ProviderResponse(
            provider_name="gemini",
            model="flash",
            text=self._out,
            usage=ProviderUsage(),
            raw=None,
        )


def test_facts_extract_system_prompt_mentions_two_and_three_sigma() -> None:
    text = _FACTS_EXTRACT_PROMPT_PATH.read_text(encoding="utf-8")
    assert f"2{_SIGMA}" in text
    assert f"3{_SIGMA}" in text


@pytest.mark.asyncio
async def test_extract_facts_packet_inserts_title_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_body = (
        "- IV / implied moves:\n"
        "- Post-Earnings IV: ~62%\n"
        f"- Forward 1{_SIGMA} Move (May 15): ±13.46% (±$17.74)\n"
        f"- Forward 2{_SIGMA} Move (May 15): ±26.92% (±$35.48)\n"
        f"- Forward 3{_SIGMA} Move (May 15): ±40.38% (±$53.22)\n"
    )
    stub_reg = ProviderRegistry()
    stub_reg.register("gemini", lambda **_: _StubGemini(out_text=stub_body))

    monkeypatch.setattr(
        "equity_analyst.facts_packet.ProviderRegistry.default",
        classmethod(lambda cls: stub_reg),
    )

    cfg = RunConfig.model_validate(
        {
            "symbol": "MNDY",
            "today_date": "d",
            "today_session": "s",
            "earnings_date": "e",
            "target_dates": [],
            "next_trading_day": "n",
            "followup_open_date": "f",
            "providers": ["openai"],
            "facts_packet_extractor_provider": "gemini",
            "facts_packet_extractor_model": "gemini-3-flash-preview",
            "facts_packet_max_output_tokens": 512,
        },
    )
    text = await extract_facts_packet(synthesis_text="Some synthesis", symbol="MNDY", config=cfg)
    assert "# Market facts (frozen from iteration 1)" in text
    assert f"Forward 2{_SIGMA} Move (May 15): ±26.92% (±$35.48)" in text
    assert f"Forward 3{_SIGMA} Move (May 15): ±40.38% (±$53.22)" in text


def test_facts_packet_fallback_markdown_lists_three_sigma_unknown_rows() -> None:
    md = _facts_packet_fallback_markdown(
        reason_bullet="- Extraction timed out; treat facts as unknown.",
    )
    assert f"Forward 2{_SIGMA} Move: unknown" in md
    assert f"Forward 3{_SIGMA} Move: unknown" in md
    assert "Post-Earnings IV: unknown" in md
