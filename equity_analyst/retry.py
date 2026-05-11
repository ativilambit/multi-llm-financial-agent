from __future__ import annotations

import asyncio
import email.utils
import logging
import random
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import TypeVar

import httpx

logger = logging.getLogger(__name__)

T = TypeVar("T")

try:
    from google.genai import errors as _genai_errors

    _GENAI_ERROR_TYPES: tuple[type[BaseException], ...] = (_genai_errors.APIError,)
except Exception:  # pragma: no cover - optional dependency
    _GENAI_ERROR_TYPES = ()


def _retryable_types() -> tuple[type[BaseException], ...]:
    types_list: list[type[BaseException]] = [
        asyncio.TimeoutError,
        TimeoutError,
        httpx.ReadTimeout,
        httpx.ConnectTimeout,
        httpx.RemoteProtocolError,
    ]
    try:
        from anthropic._exceptions import (
            APIConnectionError as AnthropicAPIConnectionError,
        )
        from anthropic._exceptions import (
            APITimeoutError as AnthropicAPITimeoutError,
        )
        from anthropic._exceptions import (
            InternalServerError as AnthropicInternalServerError,
        )
        from anthropic._exceptions import (
            OverloadedError as AnthropicOverloadedError,
        )
        from anthropic._exceptions import (
            RateLimitError as AnthropicRateLimitError,
        )
        from anthropic._exceptions import (
            ServiceUnavailableError as AnthropicServiceUnavailableError,
        )

        types_list.extend(
            [
                AnthropicRateLimitError,
                AnthropicServiceUnavailableError,
                AnthropicInternalServerError,
                AnthropicOverloadedError,
                AnthropicAPITimeoutError,
                AnthropicAPIConnectionError,
            ]
        )
    except ImportError:
        pass
    try:
        import openai

        types_list.extend(
            [
                openai.RateLimitError,
                openai.InternalServerError,
                openai.APIConnectionError,
                openai.APITimeoutError,
            ]
        )
    except ImportError:
        pass
    return tuple(types_list)


_RETRYABLE = _retryable_types()

_RETRYABLE_STATUS = frozenset({429, 502, 503, 504})


def is_retryable_exception(exc: BaseException) -> bool:
    if isinstance(exc, _RETRYABLE):
        return True
    if _GENAI_ERROR_TYPES and isinstance(exc, _GENAI_ERROR_TYPES):
        code = getattr(exc, "code", None)
        if code in _RETRYABLE_STATUS:
            return True
        resp = getattr(exc, "response", None)
        resp_status = getattr(resp, "status_code", None) if resp is not None else None
        return resp_status in _RETRYABLE_STATUS
    status = getattr(exc, "status_code", None)
    return status in _RETRYABLE_STATUS


def _parse_retry_after_seconds(value: str) -> float | None:
    try:
        sec = float(value.strip())
        if sec > 0:
            return min(60.0, sec)
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(value)
            if when.tzinfo is None:
                when = when.replace(tzinfo=UTC)
            delta = (when - datetime.now(UTC)).total_seconds()
            if delta > 0:
                return float(min(60.0, delta))
        except (TypeError, ValueError, OSError):
            return None
    return None


def _parse_google_duration_seconds(value: str) -> float | None:
    s = value.strip()
    if s.endswith("s") and len(s) > 1:
        try:
            sec = float(s[:-1])
            if sec > 0:
                return min(60.0, sec)
        except ValueError:
            return None
    return None


def _retry_after_seconds_from_genai_details(details: object) -> float | None:
    """Parse RetryInfo-style ``retryDelay`` (e.g. ``\"3s\"``) from GenAI error JSON."""
    if not isinstance(details, dict):
        return None
    blocks: list[object] = []
    err = details.get("error")
    if isinstance(err, dict):
        dlist = err.get("details")
        if isinstance(dlist, list):
            blocks.extend(dlist)
    top = details.get("details")
    if isinstance(top, list):
        blocks.extend(top)
    for item in blocks:
        if not isinstance(item, dict):
            continue
        rd = item.get("retryDelay")
        if rd is None:
            continue
        if isinstance(rd, str):
            parsed = _parse_google_duration_seconds(rd)
            if parsed is not None:
                return parsed
        if isinstance(rd, dict):
            sec_obj = rd.get("seconds")
            nanos = rd.get("nanos", 0)
            if isinstance(sec_obj, int) and sec_obj >= 0:
                total = float(sec_obj)
                if isinstance(nanos, int) and nanos > 0:
                    total += nanos / 1_000_000_000.0
                if total > 0:
                    return float(min(60.0, total))
    return None


def retry_after_seconds_from_exception(exc: BaseException) -> float | None:
    resp = getattr(exc, "response", None)
    if resp is not None:
        headers = getattr(resp, "headers", None)
        if headers is not None:
            raw = headers.get("retry-after") or headers.get("Retry-After")
            if raw:
                parsed = _parse_retry_after_seconds(str(raw))
                if parsed is not None:
                    return parsed
    ra = getattr(exc, "retry_after", None)
    if isinstance(ra, (int, float)) and ra > 0:
        return float(min(60.0, float(ra)))
    rdelay = getattr(exc, "retry_delay", None)
    if isinstance(rdelay, (int, float)) and rdelay > 0:
        return float(min(60.0, float(rdelay)))
    details_sleep = _retry_after_seconds_from_genai_details(getattr(exc, "details", None))
    if details_sleep is not None:
        return details_sleep
    return None


def _sleep_seconds(*, attempt: int, base_delay_s: float, exc: BaseException) -> float:
    header_sleep = retry_after_seconds_from_exception(exc)
    if header_sleep is not None:
        return float(header_sleep + random.uniform(0, 0.5))
    exp = min(60.0, float(base_delay_s) * (2**attempt))
    return float(exp + random.uniform(0, 0.5))


async def async_retry_call(
    factory: Callable[[], Awaitable[T]],
    *,
    provider: str,
    max_attempts: int,
    base_delay_s: float,
) -> T:
    for attempt in range(max_attempts):
        try:
            return await factory()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if attempt >= max_attempts - 1 or not is_retryable_exception(exc):
                raise
            sleep_s = _sleep_seconds(attempt=attempt, base_delay_s=base_delay_s, exc=exc)
            logger.info(
                "retrying provider=%s attempt=%s/%s reason=%s sleep_s=%.2f",
                provider,
                attempt + 2,
                max_attempts,
                type(exc).__name__,
                sleep_s,
            )
            await asyncio.sleep(sleep_s)
    raise RuntimeError("async_retry_call: exhausted attempts without return or raise")
