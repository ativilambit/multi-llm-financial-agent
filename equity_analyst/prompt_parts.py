from __future__ import annotations

from pathlib import Path
from typing import Any

from equity_analyst.synthesizer_blend import SYNTHESIZER_T0_BLEND_PLACEHOLDER

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_PROMPT_FILE_CACHE: dict[str, str] = {}


def _invariants_markdown_for_synthesizer_prepend() -> str:
    inv_path = _PROMPTS_DIR / "policy" / "invariants.md"
    return (
        inv_path.read_text(encoding="utf-8")
        .rstrip()
        .replace("{{ t0_blend_literal }}", SYNTHESIZER_T0_BLEND_PLACEHOLDER)
    )


def _load_prompt_file(name: str) -> str:
    if name not in _PROMPT_FILE_CACHE:
        path = _PROMPTS_DIR / name
        body = path.read_text(encoding="utf-8").rstrip()
        if name == "synthesizer_system.md":
            body = f"{_invariants_markdown_for_synthesizer_prepend()}\n\n{body}"
        _PROMPT_FILE_CACHE[name] = body
    return _PROMPT_FILE_CACHE[name]


def __getattr__(name: str) -> str:
    if name == "EQUITY_ANALYST_SYSTEM_PROMPT":
        return _load_prompt_file("equity_analyst_system.md")
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def ephemeral_cache_control(*, ttl_1h: bool = True) -> dict[str, Any]:
    """Anthropic prompt cache breakpoint; 1h TTL reduces churn for repeated template runs."""
    if ttl_1h:
        return {"type": "ephemeral", "ttl": "1h"}
    return {"type": "ephemeral"}
