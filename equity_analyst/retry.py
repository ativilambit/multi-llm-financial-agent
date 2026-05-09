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


def is_retryable_exception(exc: BaseException) -> bool:
    if isinstance(exc, _RETRYABLE):
        return True
    status = getattr(exc, "status_code", None)
    return status in (429, 503, 502, 504)


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
