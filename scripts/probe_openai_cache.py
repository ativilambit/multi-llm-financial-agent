#!/usr/bin/env python3
"""Two back-to-back OpenAI Responses calls to verify automatic prompt caching (usage.cached_tokens)."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


async def _main() -> int:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from equity_analyst.prompt_parts import EQUITY_ANALYST_SYSTEM_PROMPT
    from equity_analyst.providers.openai_provider import (
        OpenAIProvider,
        _prompt_cache_read_tokens,
        _responses_input_messages,
        _serialize_responses_request_body_for_debug,
    )

    load_dotenv(root / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        print("Missing OPENAI_API_KEY; set it or add to .env", file=sys.stderr)
        return 1

    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.INFO)

    model = os.environ.get("PROBE_OPENAI_MODEL", "gpt-5.5")
    static = EQUITY_ANALYST_SYSTEM_PROMPT
    user = (
        "Reply with exactly one word: OK. "
        "This line is only to separate static vs user blocks for the probe; do not elaborate."
    )
    full = f"{static}\n\n{user}"

    p = OpenAIProvider(model=model)
    max_out = int(os.environ.get("PROBE_MAX_OUTPUT_TOKENS", "64"))

    async def one_call(n: int) -> None:
        r = await p.generate(
            full,
            enable_web_search=True,
            max_output_tokens=max_out,
            cacheable_prefix=static,
            user_message_for_cache=user,
        )
        raw = r.raw
        usage = getattr(raw, "usage", None) if raw is not None else None
        cached = _prompt_cache_read_tokens(usage)
        in_tok = getattr(usage, "input_tokens", None) if usage is not None else None
        print(f"call_{n}: cached_tokens={cached} input_tokens={in_tok} latency_s={r.latency_s:.3f}")

    await one_call(1)
    await one_call(2)

    payload = _responses_input_messages(cacheable_prefix=static, user_message_for_cache=user)
    tools = [{"type": "web_search"}]
    body = _serialize_responses_request_body_for_debug(input_payload=payload, tools=tools)
    h = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
    print(f"stable_prefix_hash_sha256_16={h} (identical for both calls with same static/tools/model)")
    print("If call_1 cached_tokens=0 and call_2 > 0, automatic caching is working for this shape.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
