from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
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

_GOOD_MARKDOWN = f"""# Market facts (frozen from iteration 1)

- Last verified close: $97.02 (Mon)
- Session range: $96 - $99
- PCR: not stated
- Short interest: 7% of float
- IV / implied moves:
  - Post-Earnings IV: ~62%
  - Forward 1{_SIGMA} Move (May 12): ±13.2% (±$11.22)
  - Forward 2{_SIGMA} Move (May 12): ±26.4% (±$22.44)
  - Forward 3{_SIGMA} Move (May 12): ±39.6% (±$33.66)
- Analyst targets: consensus $105 (n=12)
- Historical Earnings Reactions: mixed
- Key Qualitative Anchors: supply chain

"""

_BAD_TRUNCATED = f"""# Market facts (frozen from iteration 1)

±$11.22)
       - Forward 2{_SIGMA} Move (May 12): ±26.4% (±$22.44)
       - Forward 3{_SIGMA} Move (May 12): ±39.6% (±$33.66)
    6. Session SD targets:
       Tue May"""


def _gemini_max_tokens_raw() -> Any:
    fr = SimpleNamespace(name="MAX_TOKENS")
    return SimpleNamespace(candidates=[SimpleNamespace(finish_reason=fr)])


class _StubFactsExtractor:
    name = "gemini"

    def __init__(
        self,
        outputs: list[tuple[str, Any | None]],
        *,
        expect_symbol: str = "MNDY",
    ) -> None:
        self._outputs = outputs
        self._call = 0
        self.max_output_tokens_per_call: list[int | None] = []
        self.expect_symbol = expect_symbol

    async def generate(
        self,
        prompt: str,
        *,
        enable_web_search: bool = True,
        max_output_tokens: int | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        assert self.expect_symbol in prompt
        assert enable_web_search is False
        self.max_output_tokens_per_call.append(max_output_tokens)
        idx = min(self._call, len(self._outputs) - 1)
        text, raw = self._outputs[idx]
        self._call += 1
        return ProviderResponse(
            provider_name="gemini",
            model="flash",
            text=text,
            usage=ProviderUsage(),
            raw=raw,
        )


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


def test_facts_extract_system_prompt_mentions_two_and_three_sigma() -> None:
    text = _FACTS_EXTRACT_PROMPT_PATH.read_text(encoding="utf-8")
    assert f"2{_SIGMA}" in text
    assert f"3{_SIGMA}" in text


def _facts_cfg(**kwargs: Any) -> RunConfig:
    base: dict[str, Any] = {
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
    }
    base.update(kwargs)
    return RunConfig.model_validate(base)


@pytest.mark.asyncio
async def test_extract_facts_packet_success_no_retry_no_truncation_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    stub = _StubFactsExtractor([(_GOOD_MARKDOWN, None)])
    stub_reg = ProviderRegistry()
    stub_reg.register("gemini", lambda **_: stub)
    monkeypatch.setattr(
        "equity_analyst.facts_packet.ProviderRegistry.default",
        classmethod(lambda cls: stub_reg),
    )
    cfg = _facts_cfg()
    with caplog.at_level(logging.INFO, logger="equity_analyst.facts_packet"):
        text = await extract_facts_packet(synthesis_text="Some synthesis", symbol="MNDY", config=cfg)
    assert "# Market facts (frozen from iteration 1)" in text
    assert f"Forward 3{_SIGMA} Move (May 12): ±39.6% (±$33.66)" in text
    assert text.endswith("\n")
    assert "looks truncated/malformed" not in caplog.text
    assert "Facts packet extracted chars=" in caplog.text
    assert stub.max_output_tokens_per_call == [512]


@pytest.mark.asyncio
async def test_extract_facts_packet_truncated_retries_then_fallback(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    raw = _gemini_max_tokens_raw()
    stub = _StubFactsExtractor([(_BAD_TRUNCATED, raw), (_BAD_TRUNCATED, raw)])
    stub_reg = ProviderRegistry()
    stub_reg.register("gemini", lambda **_: stub)
    monkeypatch.setattr(
        "equity_analyst.facts_packet.ProviderRegistry.default",
        classmethod(lambda cls: stub_reg),
    )
    cfg = _facts_cfg(facts_packet_max_output_tokens=512)
    with caplog.at_level(logging.WARNING, logger="equity_analyst.facts_packet"):
        text = await extract_facts_packet(synthesis_text="Some synthesis", symbol="MNDY", config=cfg)
    assert "facts_packet: output_chars=" in caplog.text
    assert "looks truncated/malformed" in caplog.text
    assert "treat facts as unknown" in text or "remained malformed after retry" in text
    assert stub.max_output_tokens_per_call == [512, 1024]


@pytest.mark.asyncio
async def test_extract_facts_packet_truncated_retries_on_heuristic_then_succeeds(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Bad shape alone (no MAX_TOKENS signal) still triggers one doubled-budget retry."""
    stub = _StubFactsExtractor([(_BAD_TRUNCATED, None), (_GOOD_MARKDOWN, None)])
    stub_reg = ProviderRegistry()
    stub_reg.register("gemini", lambda **_: stub)
    monkeypatch.setattr(
        "equity_analyst.facts_packet.ProviderRegistry.default",
        classmethod(lambda cls: stub_reg),
    )
    cfg = _facts_cfg(facts_packet_max_output_tokens=512)
    with caplog.at_level(logging.WARNING, logger="equity_analyst.facts_packet"):
        text = await extract_facts_packet(synthesis_text="Some synthesis", symbol="MNDY", config=cfg)
    assert "facts_packet: output_chars=" in caplog.text
    assert "looks truncated/malformed" in caplog.text
    assert f"Forward 3{_SIGMA} Move (May 12): ±39.6% (±$33.66)" in text
    assert stub.max_output_tokens_per_call == [512, 1024]


def test_facts_packet_fallback_markdown_lists_three_sigma_unknown_rows() -> None:
    md = _facts_packet_fallback_markdown(
        reason_bullet="- Extraction timed out; treat facts as unknown.",
    )
    assert f"Forward 2{_SIGMA} Move: unknown" in md
    assert f"Forward 3{_SIGMA} Move: unknown" in md
    assert "Post-Earnings IV: unknown" in md
