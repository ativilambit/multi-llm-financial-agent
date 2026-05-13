from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from equity_analyst.config import ProviderConfig, RunConfig
from equity_analyst.db_ops import best_effort_upsert_run_and_responses
from equity_analyst.drive_uploader import (
    log_drive_upload_plan_from_config,
    maybe_upload_run_to_drive,
)
from equity_analyst.gemini_cache import GeminiCacheIndex
from equity_analyst.logging_setup import attach_run_file_logging
from equity_analyst.pdf_writer import maybe_write_pdf_sibling
from equity_analyst.prompt_export import prompt_call_context, use_prompt_exporter
from equity_analyst.prompting import render_prompt, split_static_dynamic
from equity_analyst.provider_runtime import (
    effective_synthesizer_web_search,
    effective_web_search,
    failure_response,
    failure_response_from_completed,
    fan_out_max_output_tokens,
    partition_provider_responses,
    provider_timeout_s,
    run_error_record,
)
from equity_analyst.providers.anthropic_provider import AnthropicProvider
from equity_analyst.providers.gemini_provider import GeminiProvider
from equity_analyst.providers.openai_provider import OpenAIProvider
from equity_analyst.providers.registry import ProviderRegistry
from equity_analyst.retry import async_retry_call
from equity_analyst.synthesizer import (
    SynthesisResult,
    Synthesizer,
    format_synthesis_artifact_markdown,
)
from equity_analyst.types import ProviderResponse

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunArtifacts:
    output_dir: Path
    provider_files: dict[str, Path]
    synthesis_file: Path
    run_json: Path


def _provider_output_filename(provider_name: str) -> str:
    if provider_name == "anthropic":
        return "claude.md"
    if provider_name == "openai":
        return "openai.md"
    if provider_name == "gemini":
        return "gemini.md"
    if provider_name == "grok":
        return "grok.md"
    return f"{provider_name}.md"


class Orchestrator:
    def __init__(self, *, config: RunConfig, prompt_path: Path | None = None) -> None:
        self._config = config
        self._prompt_path = prompt_path or Path("prompts/equity_analyst.j2")
        self._registry = ProviderRegistry.default()

    def _make_output_dir(self) -> Path:
        ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
        out = Path("outputs") / f"{self._config.symbol}_{ts}"
        out.mkdir(parents=True, exist_ok=False)
        return out

    async def run_async(
        self, *, dry_run: bool, enable_web_search: bool = True
    ) -> tuple[str, RunArtifacts]:
        rendered = render_prompt(self._config, self._prompt_path)
        out_dir = self._make_output_dir()
        started_at_utc = datetime.now(tz=UTC).replace(microsecond=0)
        attach_run_file_logging(out_dir / "agent.log")
        self._config = log_drive_upload_plan_from_config(self._config)

        names = self._config.provider_names()
        provider_files: dict[str, Path] = {p: out_dir / _provider_output_filename(p) for p in names}
        synthesis_file = out_dir / "synthesis.md"
        run_json = out_dir / "run.json"

        artifacts = RunArtifacts(
            output_dir=out_dir,
            provider_files=provider_files,
            synthesis_file=synthesis_file,
            run_json=run_json,
        )

        logger.info(
            "Run start symbol=%s providers=%s synthesizer=%s dry_run=%s output_dir=%s web_search=%s",
            self._config.symbol,
            names,
            self._config.synthesizer.name,
            dry_run,
            str(out_dir.resolve()),
            enable_web_search,
        )

        if dry_run:
            preview_lines = [
                "# DRY RUN (no API calls made)",
                "",
                f"Symbol: {self._config.symbol}",
                f"Providers: {', '.join(names)}",
                f"Synthesizer: {self._config.synthesizer.name}",
                f"Template: {rendered.template_path}",
                f"Web search (run default): {enable_web_search}",
                "",
                "## Rendered prompt",
                rendered.text.rstrip(),
                "",
            ]
            run_json.write_text(
                json.dumps(
                    {
                        "dry_run": True,
                        "run_profile": self._config.run_profile,
                        "started_at_utc": started_at_utc.isoformat(),
                        "finished_at_utc": datetime.now(tz=UTC).replace(microsecond=0).isoformat(),
                        "timestamp_utc": datetime.now(tz=UTC).isoformat(),
                        "config": self._config.model_dump(),
                        "template_path": rendered.template_path,
                        "providers": {
                            pc.name: {
                                "enabled": True,
                                "web_search": effective_web_search(
                                    run_default=enable_web_search, pc=pc
                                ),
                            }
                            for pc in self._config.providers
                        },
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            preview_md = "\n".join(preview_lines)
            synthesis_file.write_text(preview_md, encoding="utf-8")
            maybe_write_pdf_sibling(
                pdf_output_enabled=self._config.pdf_output_enabled,
                md_path=synthesis_file,
                markdown_text=preview_md,
            )
            if self._config.drive_upload_enabled:
                await maybe_upload_run_to_drive(
                    self._config, out_dir, append_synthesis_footer=False
                )
            finished_at_utc = datetime.now(tz=UTC).replace(microsecond=0)
            try:
                run_json_data = json.loads(run_json.read_text(encoding="utf-8"))
            except Exception:
                run_json_data = {}
            with contextlib.suppress(Exception):
                await best_effort_upsert_run_and_responses(
                    cfg=self._config,
                    run_dir=out_dir,
                    run_json_data=run_json_data,
                    started_at_utc=started_at_utc,
                    finished_at_utc=finished_at_utc,
                    provider_responses=[],
                    synthesis_path=synthesis_file,
                    database_url=self._config.database_url,
                )
            logger.info("Run end (dry-run) output_dir=%s", str(out_dir.resolve()))
            return ("\n".join(preview_lines), artifacts)

        live_t0 = time.perf_counter()
        run_errors: list[dict[str, Any]] = []
        gemini_cache_index: GeminiCacheIndex | None = (
            GeminiCacheIndex() if self._config.prompt_cache_enabled else None
        )

        fan_ctx = copy.deepcopy(rendered.context)
        with use_prompt_exporter(out_dir):
            static_block, user_block = split_static_dynamic(rendered)

            async def _heartbeat(stop: asyncio.Event, provider_names: list[str]) -> None:
                start = time.perf_counter()
                while True:
                    try:
                        await asyncio.wait_for(stop.wait(), timeout=30.0)
                        return
                    except TimeoutError:
                        logger.info(
                            "Still waiting on providers=%s (%ss elapsed)",
                            provider_names,
                            int(time.perf_counter() - start),
                        )

            async def _run_one(pc: ProviderConfig) -> ProviderResponse:
                t0 = time.perf_counter()
                provider = self._registry.create(
                    pc.name,
                    model=pc.model,
                    gemini_cache_index=gemini_cache_index,
                    gemini_cache_ttl_s=self._config.gemini_cache_ttl_s,
                )
                ws = effective_web_search(run_default=enable_web_search, pc=pc)
                timeout_s = provider_timeout_s(pc, self._config)

                async def _attempt() -> ProviderResponse:
                    mot = fan_out_max_output_tokens(pc, self._config.max_output_tokens)
                    if isinstance(provider, AnthropicProvider):
                        return await provider.generate(
                            rendered.text,
                            enable_web_search=ws,
                            max_output_tokens=mot,
                            prompt_cache_enabled=self._config.prompt_cache_enabled,
                            user_message_for_cache=user_block,
                            force_tool_use=self._config.anthropic_force_tool_use,
                        )
                    if isinstance(provider, GeminiProvider) and self._config.prompt_cache_enabled:
                        return await provider.generate(
                            rendered.text,
                            enable_web_search=ws,
                            max_output_tokens=mot,
                            cacheable_prefix=static_block,
                            user_message_for_cache=user_block,
                        )
                    if isinstance(provider, OpenAIProvider):
                        return await provider.generate(
                            rendered.text,
                            enable_web_search=ws,
                            max_output_tokens=mot,
                            cacheable_prefix=static_block,
                            user_message_for_cache=user_block,
                        )
                    return await provider.generate(
                        rendered.text,
                        enable_web_search=ws,
                        max_output_tokens=mot,
                    )

                try:
                    return await asyncio.wait_for(
                        async_retry_call(
                            _attempt,
                            provider=pc.name,
                            max_attempts=self._config.retry_max_attempts,
                            base_delay_s=self._config.retry_base_delay_s,
                        ),
                        timeout=timeout_s,
                    )
                except asyncio.CancelledError:
                    raise
                except TimeoutError as exc:
                    return failure_response_from_completed(pc.name, exc, started_perf=t0)

            with prompt_call_context(node="fan_out", analyst_render_context=fan_ctx):
                logger.info("Starting provider generation providers=%s", names)
                stop_hb = asyncio.Event()
                hb = asyncio.create_task(_heartbeat(stop_hb, names))
                batch_t0 = time.perf_counter()
                try:
                    responses_list = await asyncio.gather(
                        *[_run_one(pc) for pc in self._config.providers],
                        return_exceptions=True,
                    )
                finally:
                    stop_hb.set()
                    hb.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await hb

                responses: dict[str, ProviderResponse] = {}
                for pc, item in zip(self._config.providers, responses_list, strict=True):
                    if isinstance(item, ProviderResponse):
                        responses[pc.name] = item
                    elif isinstance(item, Exception):
                        responses[pc.name] = failure_response(pc.name, item, latency_s=None)
                    else:
                        raise item

                parallel_batch_wall_s = time.perf_counter() - batch_t0

            for name, resp in responses.items():
                logger.info(
                    "Provider finished name=%s model=%s latency_s=%s",
                    name,
                    resp.model,
                    f"{resp.latency_s:.3f}" if resp.latency_s is not None else "n/a",
                )

            for name, resp in responses.items():
                provider_files[name].write_text(resp.text.rstrip() + "\n", encoding="utf-8")

            syn_cfg = self._config.synthesizer
            synth_provider = self._registry.create(syn_cfg.name, model=syn_cfg.model)
            syn_ws = effective_synthesizer_web_search(run_default=enable_web_search, syn=syn_cfg)
            syn_timeout = self._config.synthesizer_timeout_s()
            summarize_fallback_llm = None
            fb_name = self._config.oversized_summarize_fallback_provider
            if fb_name:
                for pc in self._config.providers:
                    if pc.name == fb_name:
                        summarize_fallback_llm = self._registry.create(pc.name, model=pc.model)
                        break
            syn_t0 = time.perf_counter()
            with prompt_call_context(node="synthesize"):
                try:
                    synthesis = await asyncio.wait_for(
                        Synthesizer(synth_provider).synthesize(
                            original_prompt=rendered.text,
                            responses=responses,
                            enable_web_search=syn_ws,
                            max_output_tokens=self._config.synthesizer_max_output_tokens,
                            synthesizer_max_input_tokens=self._config.synthesizer_max_input_tokens,
                            retry_max_attempts=self._config.retry_max_attempts,
                            retry_base_delay_s=self._config.retry_base_delay_s,
                            anthropic_force_tool_use=self._config.anthropic_force_tool_use,
                            symbol=self._config.symbol,
                            summarize_oversized_providers=self._config.summarize_oversized_providers,
                            summarize_threshold_input_tokens=self._config.summarize_threshold_input_tokens,
                            oversized_summarize_provider=self._config.oversized_summarize_provider,
                            oversized_summarize_model=self._config.oversized_summarize_model,
                            oversized_summarize_max_output_tokens=self._config.oversized_summarize_max_output_tokens,
                            oversized_summarize_max_input_tokens=self._config.oversized_summarize_max_input_tokens,
                            oversized_summarize_min_retention=self._config.oversized_summarize_min_retention,
                            oversized_summarize_fallback_provider=summarize_fallback_llm,
                            computed_sigma_bands_markdown=(
                                str(rendered.context.get("computed_sigma_bands_markdown") or "").strip()
                                or None
                            ),
                            t0_blend_preset=self._config.t0_blend_preset,
                        ),
                        timeout=syn_timeout,
                    )
                except asyncio.CancelledError:
                    raise
                except TimeoutError as exc:
                    logger.error(
                        "Synthesis failed: provider=%s error_type=%s detail=%r",
                        syn_cfg.name,
                        type(exc).__name__,
                        exc,
                    )
                    run_errors.append(run_error_record(stage="synthesis", provider=syn_cfg.name, exc=exc))
                    synthesis_resp = failure_response_from_completed(
                        syn_cfg.name,
                        exc,
                        started_perf=syn_t0,
                    )
                    synthesis = SynthesisResult(
                        response=synthesis_resp, prompt="(synthesis stage timed out)"
                    )
                except Exception as exc:
                    logger.error(
                        "Synthesis failed: provider=%s error_type=%s detail=%r",
                        syn_cfg.name,
                        type(exc).__name__,
                        exc,
                    )
                    run_errors.append(run_error_record(stage="synthesis", provider=syn_cfg.name, exc=exc))
                    synthesis_resp = failure_response_from_completed(
                        syn_cfg.name,
                        exc,
                        started_perf=syn_t0,
                    )
                    synthesis = SynthesisResult(
                        response=synthesis_resp,
                        prompt=f"(synthesis exception: {type(exc).__name__})",
                    )
            syn_wall_s = time.perf_counter() - syn_t0

            _, failed_only = partition_provider_responses(responses)
            if synthesis.response.model == "error:AllProvidersFailed":
                run_errors.append(
                    {
                        "stage": "synthesis",
                        "provider": syn_cfg.name,
                        "error_type": "AllProvidersFailed",
                        "detail": f"excluded_failed_providers={sorted(failed_only)}",
                    }
                )

            synthesis_md = format_synthesis_artifact_markdown(synthesis=synthesis, responses=responses)
            synthesis_file.write_text(synthesis_md, encoding="utf-8")
            maybe_write_pdf_sibling(
                pdf_output_enabled=self._config.pdf_output_enabled,
                md_path=synthesis_file,
                markdown_text=synthesis_md,
            )

            total_wall_s = time.perf_counter() - live_t0
            timing: dict[str, Any] = {
                "parallel_provider_batch_wall_s": round(parallel_batch_wall_s, 3),
                "synthesis_wall_s": round(syn_wall_s, 3),
                "total_wall_s": round(total_wall_s, 3),
                "per_provider": {
                    n: {
                        "latency_s": responses[n].latency_s,
                        "model": responses[n].model,
                    }
                    for n in names
                },
            }

            run_meta: dict[str, Any] = {
                "dry_run": False,
                "run_profile": self._config.run_profile,
                "started_at_utc": started_at_utc.isoformat(),
                "finished_at_utc": datetime.now(tz=UTC).replace(microsecond=0).isoformat(),
                "timestamp_utc": datetime.now(tz=UTC).isoformat(),
                "config": self._config.model_dump(),
                "template_path": rendered.template_path,
                "timing": timing,
                "providers": {
                    name: {
                        "provider_name": resp.provider_name,
                        "model": resp.model,
                        "usage": asdict(resp.usage),
                        "latency_s": resp.latency_s,
                    }
                    for name, resp in responses.items()
                },
                "synthesis": {
                    "provider": synthesis.response.provider_name,
                    "model": synthesis.response.model,
                    "usage": asdict(synthesis.response.usage),
                    "latency_s": synthesis.response.latency_s,
                },
                "errors": run_errors,
                "options_chain_data": rendered.context.get("options_chain_data") or {},
                "computed_sigma_bands_table": (
                    rendered.context.get("computed_sigma_bands_table")
                    if isinstance(rendered.context.get("computed_sigma_bands_table"), dict)
                    else None
                ),
            }
            run_json.write_text(json.dumps(run_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            if self._config.drive_upload_enabled:
                await maybe_upload_run_to_drive(self._config, out_dir, append_synthesis_footer=False)

            finished_at_utc = datetime.now(tz=UTC).replace(microsecond=0)
            try:
                run_json_data = json.loads(run_json.read_text(encoding="utf-8"))
            except Exception:
                run_json_data = {}
            with contextlib.suppress(Exception):
                provider_rows: list[tuple[int | None, str, ProviderResponse, str, bool]] = []
                for pc in self._config.providers:
                    p_resp = responses.get(pc.name)
                    if p_resp is None:
                        continue
                    p = provider_files[pc.name]
                    try:
                        response_rel = str(p.relative_to(out_dir.parent))
                    except Exception:
                        response_rel = str(p)
                    provider_rows.append(
                        (
                            None,
                            pc.name,
                            p_resp,
                            response_rel,
                            effective_web_search(run_default=enable_web_search, pc=pc),
                        )
                    )
                # synthesizer row (points to synthesis.md)
                try:
                    syn_rel = str(synthesis_file.relative_to(out_dir.parent))
                except Exception:
                    syn_rel = str(synthesis_file)
                provider_rows.append(
                    (
                        None,
                        synthesis.response.provider_name,
                        synthesis.response,
                        syn_rel,
                        effective_synthesizer_web_search(
                            run_default=enable_web_search, syn=self._config.synthesizer
                        ),
                    )
                )
                await best_effort_upsert_run_and_responses(
                    cfg=self._config,
                    run_dir=out_dir,
                    run_json_data=run_json_data,
                    started_at_utc=started_at_utc,
                    finished_at_utc=finished_at_utc,
                    provider_responses=provider_rows,
                    synthesis_path=synthesis_file,
                    database_url=self._config.database_url,
                )

            if self._config.prediction_extract_enabled:
                from equity_analyst.prediction_extract import run_prediction_extract_for_run_dir

                await run_prediction_extract_for_run_dir(run_dir=out_dir, cfg=self._config)

            logger.info(
                "Run end (live) output_dir=%s synthesis_model=%s synthesis_latency_s=%s timing=%s",
                str(out_dir.resolve()),
                synthesis.response.model,
                f"{synthesis.response.latency_s:.3f}"
                if synthesis.response.latency_s is not None
                else "n/a",
                timing,
            )
            return (synthesis.response.text, artifacts)

    def run_sync(self, *, dry_run: bool, enable_web_search: bool = True) -> str:
        text, _artifacts = asyncio.run(
            self.run_async(dry_run=dry_run, enable_web_search=enable_web_search)
        )
        return text
