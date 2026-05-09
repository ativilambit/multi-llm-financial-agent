from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from equity_analyst.config import RunConfig
from equity_analyst.prompting import render_prompt
from equity_analyst.providers.registry import ProviderRegistry
from equity_analyst.synthesizer import Synthesizer
from equity_analyst.types import ProviderResponse


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

    async def run_async(self, *, dry_run: bool, enable_web_search: bool = True) -> tuple[str, RunArtifacts]:
        rendered = render_prompt(self._config, self._prompt_path)
        out_dir = self._make_output_dir()

        provider_files: dict[str, Path] = {
            p: out_dir / _provider_output_filename(p) for p in self._config.providers
        }
        synthesis_file = out_dir / "synthesis.md"
        run_json = out_dir / "run.json"

        artifacts = RunArtifacts(
            output_dir=out_dir,
            provider_files=provider_files,
            synthesis_file=synthesis_file,
            run_json=run_json,
        )

        if dry_run:
            preview_lines = [
                "# DRY RUN (no API calls made)",
                "",
                f"Symbol: {self._config.symbol}",
                f"Providers: {', '.join(self._config.providers)}",
                f"Synthesizer: {self._config.synthesizer}",
                f"Template: {rendered.template_path}",
                f"Web search enabled: {enable_web_search}",
                "",
                "## Rendered prompt",
                rendered.text.rstrip(),
                "",
            ]
            run_json.write_text(
                json.dumps(
                    {
                        "dry_run": True,
                        "timestamp_utc": datetime.now(tz=UTC).isoformat(),
                        "config": self._config.model_dump(),
                        "template_path": rendered.template_path,
                        "providers": {
                            p: {"enabled": True, "web_search": enable_web_search}
                            for p in self._config.providers
                        },
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            synthesis_file.write_text("\n".join(preview_lines), encoding="utf-8")
            return ("\n".join(preview_lines), artifacts)

        async def _run_one(provider_name: str) -> ProviderResponse:
            provider = self._registry.create(provider_name)
            return await provider.generate(rendered.text, enable_web_search=enable_web_search)

        responses_list = await asyncio.gather(*[_run_one(p) for p in self._config.providers])
        responses: dict[str, ProviderResponse] = {r.provider_name: r for r in responses_list}

        for name, resp in responses.items():
            provider_files[name].write_text(resp.text.rstrip() + "\n", encoding="utf-8")

        synth_provider = self._registry.create(self._config.synthesizer)
        synthesis = await Synthesizer(synth_provider).synthesize(
            original_prompt=rendered.text, responses=responses, enable_web_search=enable_web_search
        )
        synthesis_file.write_text(synthesis.response.text.rstrip() + "\n", encoding="utf-8")

        run_meta: dict[str, Any] = {
            "dry_run": False,
            "timestamp_utc": datetime.now(tz=UTC).isoformat(),
            "config": self._config.model_dump(),
            "template_path": rendered.template_path,
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
        }
        run_json.write_text(json.dumps(run_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        return (synthesis.response.text, artifacts)

    def run_sync(self, *, dry_run: bool, enable_web_search: bool = True) -> str:
        text, _artifacts = asyncio.run(
            self.run_async(dry_run=dry_run, enable_web_search=enable_web_search)
        )
        return text

