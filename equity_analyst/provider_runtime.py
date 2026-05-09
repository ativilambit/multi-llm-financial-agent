from __future__ import annotations

import time
from typing import Any

from equity_analyst.config import ProviderConfig, RunConfig, SynthesizerConfig
from equity_analyst.types import ProviderResponse, ProviderUsage


def is_failed_provider_response(resp: ProviderResponse) -> bool:
    if resp.model.startswith("error:"):
        return True
    return not resp.text.strip()


def run_error_record(*, stage: str, provider: str, exc: BaseException) -> dict[str, Any]:
    return {
        "stage": stage,
        "provider": provider,
        "error_type": type(exc).__name__,
        "detail": repr(exc),
    }


def partition_provider_responses(
    responses: dict[str, ProviderResponse],
) -> tuple[dict[str, ProviderResponse], dict[str, ProviderResponse]]:
    healthy: dict[str, ProviderResponse] = {}
    failed: dict[str, ProviderResponse] = {}
    for name, resp in responses.items():
        if is_failed_provider_response(resp):
            failed[name] = resp
        else:
            healthy[name] = resp
    return healthy, failed


def effective_web_search(*, run_default: bool, pc: ProviderConfig) -> bool:
    if pc.web_search is not None:
        return bool(pc.web_search)
    return run_default


def effective_synthesizer_web_search(*, run_default: bool, syn: SynthesizerConfig) -> bool:
    if syn.web_search is not None:
        return bool(syn.web_search)
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
        text=(
            f"# Provider `{name}` failed ({label})\n\n"
            f"```\n{exc!r}\n```\n"
        ),
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
