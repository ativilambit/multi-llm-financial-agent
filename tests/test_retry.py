from __future__ import annotations

from typing import Any

import httpx
import pytest

from equity_analyst.retry import (
    async_retry_call,
    format_retry_exception_reason,
    is_retryable_exception,
    retry_after_seconds_from_exception,
)

try:
    from google.genai import errors as genai_errors
except ImportError:  # pragma: no cover
    genai_errors = None  # type: ignore[assignment]


@pytest.mark.skipif(genai_errors is None, reason="google.genai not installed")
def test_is_retryable_genai_client_error_codes() -> None:
    req = httpx.Request("POST", "https://generativelanguage.googleapis.com/v1/models")
    for code in (429, 502, 503, 504):
        resp = httpx.Response(code, request=req)
        exc = genai_errors.ClientError(code, {"error": {}}, resp)
        assert is_retryable_exception(exc) is True
    resp400 = httpx.Response(400, request=req)
    exc400 = genai_errors.ClientError(400, {"error": {}}, resp400)
    assert is_retryable_exception(exc400) is False


@pytest.mark.skipif(genai_errors is None, reason="google.genai not installed")
def test_retry_after_from_genai_details_retry_delay_string() -> None:
    req = httpx.Request("POST", "https://example.com")
    exc = genai_errors.ClientError(
        429,
        {
            "error": {
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.RetryInfo",
                        "retryDelay": "4s",
                    }
                ]
            }
        },
        httpx.Response(429, request=req),
    )
    assert retry_after_seconds_from_exception(exc) == 4.0


def test_is_retryable_rate_limit() -> None:
    import anthropic

    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    resp = httpx.Response(429, request=req)
    exc = anthropic.RateLimitError("rl", response=resp, body=None)
    assert is_retryable_exception(exc) is True


def _anthropic_api_status_error(*, status: int, body: object) -> Any:
    import anthropic

    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    resp = httpx.Response(status, request=req)
    return anthropic.APIStatusError("x", response=resp, body=body)


def test_is_retryable_anthropic_api_status_error_overloaded_error_body() -> None:
    body = {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
    exc = _anthropic_api_status_error(status=400, body=body)
    assert is_retryable_exception(exc) is True
    assert "overloaded_error" in format_retry_exception_reason(exc)


def test_is_retryable_anthropic_api_status_error_invalid_request_not_retried() -> None:
    body = {"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}}
    exc = _anthropic_api_status_error(status=400, body=body)
    assert is_retryable_exception(exc) is False


def test_is_retryable_anthropic_api_status_error_529_status() -> None:
    exc = _anthropic_api_status_error(status=529, body=None)
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


@pytest.mark.asyncio
async def test_async_retry_overloaded_retry_after_caps_at_90s(monkeypatch: Any) -> None:
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
            body = {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
            resp = httpx.Response(529, request=req, headers={"retry-after": "120"}, json=body)
            raise anthropic.APIStatusError("Overloaded", response=resp, body=body)
        return "ok"

    await async_retry_call(factory, provider="anthropic", max_attempts=3, base_delay_s=2.0)
    assert captured
    assert 90.0 <= captured[0] <= 90.6


def test_retry_after_seconds_from_exception() -> None:
    import anthropic

    req = httpx.Request("GET", "https://example.com")
    resp = httpx.Response(429, request=req, headers={"retry-after": "3"})
    exc = anthropic.RateLimitError("m", response=resp, body=None)
    ra = retry_after_seconds_from_exception(exc)
    assert ra == 3.0
