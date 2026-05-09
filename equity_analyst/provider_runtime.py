from __future__ import annotations

import time

from equity_analyst.config import ProviderConfig, RunConfig
from equity_analyst.types import ProviderResponse, ProviderUsage


def effective_web_search(*, run_default: bool, pc: ProviderConfig) -> bool:
    if pc.web_search is not None:
        return bool(pc.web_search)
    return run_default


def provider_timeout_s(pc: ProviderConfig, cfg: RunConfig) -> float:
    if pc.request_timeout_s is not None:
        return float(pc.request_timeout_s)
    return float(cfg.request_timeout_s)


def failure_response(name: str, exc: BaseException, *, latency_s: float | None) -> ProviderResponse:
    label = "timeout" if isinstance(exc, TimeoutError) else type(exc).__name__
    model = f"error:{label}"
    return ProviderResponse(
        provider_name=name,
        model=model,
        text=(f"# Provider `{name}` failed ({label})\n\n```\n{exc!r}\n```\n"),
        usage=ProviderUsage(),
        latency_s=latency_s,
        raw=None,
    )


def failure_response_from_completed(
    name: str,
    exc: BaseException,
    *,
    started_perf: float,
) -> ProviderResponse:
    return failure_response(name, exc, latency_s=time.perf_counter() - started_perf)
