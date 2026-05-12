from __future__ import annotations

import asyncio
import contextlib
import email.utils
import json
import logging
import random
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, TypeVar

import httpx

logger = logging.getLogger(__name__)

T = TypeVar("T")

try:
    from google.genai import errors as _genai_errors

    _GENAI_ERROR_TYPES: tuple[type[BaseException], ...] = (_genai_errors.APIError,)
except Exception:  # pragma: no cover - optional dependency
    _GENAI_ERROR_TYPES = ()

AnthropicAPIStatusError: Any = None
with contextlib.suppress(ImportError):  # pragma: no cover - optional dependency
    from anthropic import APIStatusError as AnthropicAPIStatusError


_ANTHROPIC_RETRYABLE_ERROR_TYPES = frozenset(
    {
        "overloaded_error",
        "rate_limit_error",
        "api_error",
        "server_error",
        "service_unavailable_error",
    },
)


def _dict_from_anthropic_body(body: object) -> dict[str, Any] | None:
    if isinstance(body, dict):
        return body
    if isinstance(body, str):
        s = body.strip()
        if not s.startswith("{"):
            return None
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _anthropic_error_type_from_body(body: object) -> str | None:
    m = _dict_from_anthropic_body(body)
    if m is None:
        return None
    err = m.get("error")
    if isinstance(err, dict):
        t = err.get("type")
        if isinstance(t, str):
            return t
    return None


def _anthropic_request_id_from_body(body: object) -> str | None:
    m = _dict_from_anthropic_body(body)
    if m is None:
        return None
    rid = m.get("request_id")
    return rid if isinstance(rid, str) else None


def _is_anthropic_api_status_error(exc: BaseException) -> bool:
    cls = AnthropicAPIStatusError
    return cls is not None and isinstance(exc, cls)


def _anthropic_overload_like_exception(exc: BaseException) -> bool:
    if not _is_anthropic_api_status_error(exc):
        return False
    body = getattr(exc, "body", None)
    if _anthropic_error_type_from_body(body) == "overloaded_error":
        return True
    return getattr(exc, "status_code", None) == 529


def _retry_delay_cap_seconds(exc: BaseException) -> float:
    return 90.0 if _anthropic_overload_like_exception(exc) else 60.0


def format_retry_exception_reason(exc: BaseException) -> str:
    if _is_anthropic_api_status_error(exc):
        et = _anthropic_error_type_from_body(getattr(exc, "body", None))
        if et:
            return f"{type(exc).__name__}({et})"
    return type(exc).__name__


def _retry_request_id_for_log(exc: BaseException) -> str | None:
    if not _is_anthropic_api_status_error(exc):
        return None
    return _anthropic_request_id_from_body(getattr(exc, "body", None))


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

_RETRYABLE_STATUS = frozenset({429, 502, 503, 504, 529})


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
    if _is_anthropic_api_status_error(exc):
        body = getattr(exc, "body", None)
        et = _anthropic_error_type_from_body(body)
        if et in _ANTHROPIC_RETRYABLE_ERROR_TYPES:
            return True
        if et == "invalid_request_error":
            return False
    status = getattr(exc, "status_code", None)
    if status in _RETRYABLE_STATUS:
        return True
    if _is_anthropic_api_status_error(exc):
        body = getattr(exc, "body", None)
        et = _anthropic_error_type_from_body(body)
        if et is not None:
            return False
    return False


def _parse_retry_after_seconds(value: str, *, max_seconds: float = 60.0) -> float | None:
    try:
        sec = float(value.strip())
        if sec > 0:
            return min(max_seconds, sec)
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(value)
            if when.tzinfo is None:
                when = when.replace(tzinfo=UTC)
            delta = (when - datetime.now(UTC)).total_seconds()
            if delta > 0:
                return float(min(max_seconds, delta))
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
    cap = _retry_delay_cap_seconds(exc)
    resp = getattr(exc, "response", None)
    if resp is not None:
        headers = getattr(resp, "headers", None)
        if headers is not None:
            raw = headers.get("retry-after") or headers.get("Retry-After")
            if raw:
                parsed = _parse_retry_after_seconds(str(raw), max_seconds=cap)
                if parsed is not None:
                    return parsed
    ra = getattr(exc, "retry_after", None)
    if isinstance(ra, (int, float)) and ra > 0:
        return float(min(cap, float(ra)))
    rdelay = getattr(exc, "retry_delay", None)
    if isinstance(rdelay, (int, float)) and rdelay > 0:
        return float(min(cap, float(rdelay)))
    details_sleep = _retry_after_seconds_from_genai_details(getattr(exc, "details", None))
    if details_sleep is not None:
        return details_sleep
    return None


def _sleep_seconds(*, attempt: int, base_delay_s: float, exc: BaseException) -> float:
    cap = _retry_delay_cap_seconds(exc)
    header_sleep = retry_after_seconds_from_exception(exc)
    if header_sleep is not None:
        return float(header_sleep + random.uniform(0, 0.5))
    exp = min(cap, float(base_delay_s) * (2**attempt))
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
            reason = format_retry_exception_reason(exc)
            rid = _retry_request_id_for_log(exc)
            if rid:
                logger.info(
                    "retrying provider=%s attempt=%s/%s reason=%s request_id=%s sleep_s=%.2f",
                    provider,
                    attempt + 2,
                    max_attempts,
                    reason,
                    rid,
                    sleep_s,
                )
            else:
                logger.info(
                    "retrying provider=%s attempt=%s/%s reason=%s sleep_s=%.2f",
                    provider,
                    attempt + 2,
                    max_attempts,
                    reason,
                    sleep_s,
                )
            await asyncio.sleep(sleep_s)
    raise RuntimeError("async_retry_call: exhausted attempts without return or raise")
