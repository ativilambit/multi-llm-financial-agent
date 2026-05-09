from __future__ import annotations

from typing import Any

import httpx
import pytest

from equity_analyst.retry import (
    async_retry_call,
    is_retryable_exception,
    retry_after_seconds_from_exception,
)


def test_is_retryable_rate_limit() -> None:
    import anthropic

    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    resp = httpx.Response(429, request=req)
    exc = anthropic.RateLimitError("rl", response=resp, body=None)
    assert is_retryable_exception(exc) is True


@pytest.mark.asyncio
async def test_async_retry_call_retries_on_429_then_success(monkeypatch: Any) -> None:
    import anthropic

    sleeps: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)

    monkeypatch.setattr("equity_analyst.retry.asyncio.sleep", fake_sleep)

    n = 0

    async def factory() -> str:
        nonlocal n
        n += 1
        if n == 1:
            req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
            resp = httpx.Response(429, request=req)
            raise anthropic.RateLimitError("rl", response=resp, body=None)
        return "ok"

    out = await async_retry_call(factory, provider="anthropic", max_attempts=3, base_delay_s=2.0)
    assert out == "ok"
    assert n == 2
    assert len(sleeps) == 1


@pytest.mark.asyncio
async def test_async_retry_call_gives_up_after_n(monkeypatch: Any) -> None:
    import anthropic

    async def fake_sleep(_s: float) -> None:
        return None

    monkeypatch.setattr("equity_analyst.retry.asyncio.sleep", fake_sleep)

    n = 0

    async def factory() -> str:
        nonlocal n
        n += 1
        req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        resp = httpx.Response(429, request=req)
        raise anthropic.RateLimitError("rl", response=resp, body=None)

    with pytest.raises(anthropic.RateLimitError):
        await async_retry_call(factory, provider="anthropic", max_attempts=2, base_delay_s=2.0)
    assert n == 2


@pytest.mark.asyncio
async def test_async_retry_call_honors_retry_after_header(monkeypatch: Any) -> None:
    import anthropic

    captured: list[float] = []

    async def fake_sleep(s: float) -> None:
        captured.append(s)

    monkeypatch.setattr("equity_analyst.retry.asyncio.sleep", fake_sleep)

    n = 0

    async def factory() -> str:
        nonlocal n
        n += 1
        if n == 1:
            req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
            resp = httpx.Response(429, request=req, headers={"retry-after": "4"})
            raise anthropic.RateLimitError("rl", response=resp, body=None)
        return "done"

    await async_retry_call(factory, provider="x", max_attempts=3, base_delay_s=2.0)
    assert captured
    assert 4.0 <= captured[0] <= 4.6


def test_retry_after_seconds_from_exception() -> None:
    import anthropic

    req = httpx.Request("GET", "https://example.com")
    resp = httpx.Response(429, request=req, headers={"retry-after": "3"})
    exc = anthropic.RateLimitError("m", response=resp, body=None)
    ra = retry_after_seconds_from_exception(exc)
    assert ra == 3.0
