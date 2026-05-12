from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_EXPORT_ENV = "EXPORT_PROMPTS"


def prompts_export_enabled() -> bool:
    """When ``EXPORT_PROMPTS=0`` (case-insensitive), skip all prompt exports."""
    return os.environ.get(_EXPORT_ENV, "1").strip().lower() not in {"0", "false", "no", "off"}


def _sanitize_model_for_filename(model: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", model.strip())
    return s[:120] if s else "model"


def _markdown_fence(body: str) -> str:
    longest = 3
    for m in re.finditer(r"`+", body):
        longest = max(longest, len(m.group(0)) + 1)
    fence = "`" * longest
    return f"{fence}\n{body}\n{fence}"


@dataclass(frozen=True)
class PromptCallMeta:
    """Per-request metadata for prompt export (from contextvars)."""

    node: str
    iteration: int | None = None
    analyst_render_context: dict[str, Any] | None = None


_prompt_exporter_ctx: ContextVar[PromptExporter | None] = ContextVar(
    "equity_analyst_prompt_exporter", default=None
)
_prompt_call_meta_ctx: ContextVar[PromptCallMeta | None] = ContextVar(
    "equity_analyst_prompt_call_meta", default=None
)
_logical_split_ctx: ContextVar[tuple[str, str] | None] = ContextVar(
    "equity_analyst_prompt_logical_split", default=None
)


def current_prompt_exporter() -> PromptExporter | None:
    return _prompt_exporter_ctx.get()


def current_prompt_call_meta() -> PromptCallMeta | None:
    return _prompt_call_meta_ctx.get()


@contextmanager
def logical_prompt_split(system: str, user: str) -> Iterator[None]:
    """Override exported system/user (e.g. combined-string APIs with a logical split)."""
    token = _logical_split_ctx.set((system, user))
    try:
        yield
    finally:
        _logical_split_ctx.reset(token)


@contextmanager
def prompt_call_context(
    *,
    node: str,
    iteration: int | None = None,
    analyst_render_context: dict[str, Any] | None = None,
) -> Iterator[None]:
    ar: dict[str, Any] | None = None
    if analyst_render_context is not None:
        ar = json.loads(json.dumps(analyst_render_context, default=str))
    meta = PromptCallMeta(
        node=node,
        iteration=iteration,
        analyst_render_context=ar,
    )
    token = _prompt_call_meta_ctx.set(meta)
    try:
        yield
    finally:
        _prompt_call_meta_ctx.reset(token)


@contextmanager
def use_prompt_exporter(run_dir: Path) -> Iterator[PromptExporter | None]:
    """Attach a ``PromptExporter`` for this run; writes ``prompts_index.md`` on exit."""
    if not prompts_export_enabled():
        t = _prompt_exporter_ctx.set(None)
        try:
            yield None
        finally:
            _prompt_exporter_ctx.reset(t)
        return
    exp = PromptExporter(run_dir)
    t = _prompt_exporter_ctx.set(exp)
    try:
        yield exp
    finally:
        exp.finalize_index()
        _prompt_exporter_ctx.reset(t)


def export_prompt(
    run_dir: Path,
    *,
    node: str,
    iteration: int | None,
    provider: str,
    model: str,
    system: str,
    user: str,
    config: Mapping[str, Any],
    sequence: int,
    context_sidecar: dict[str, Any] | None = None,
) -> Path:
    """Write one prompt artifact under ``run_dir/prompts/`` (used by tests and ``PromptExporter``)."""
    prompts_dir = run_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    prov_slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", provider)[:48] or "provider"
    mod_slug = _sanitize_model_for_filename(model)
    base = f"{sequence:04d}_{node}_{prov_slug}_{mod_slug}"
    path = prompts_dir / f"{base}.md"
    ts = datetime.now(tz=UTC).replace(microsecond=0).isoformat()
    it_s = "n/a" if iteration is None else str(iteration)
    meta_lines = [
        f"- sequence: {sequence:04d}",
        f"- timestamp: {ts}",
    ]
    for key in sorted(config.keys()):
        meta_lines.append(f"- {key}: {config[key]!r}")
    meta_block = "\n".join(meta_lines)
    body = (
        f"# {node} · iteration {it_s} · provider {provider} · model {model}\n\n"
        f"## Metadata\n{meta_block}\n\n"
        f"## System message\n\n{_markdown_fence(system)}\n\n"
        f"## User message\n\n{_markdown_fence(user)}\n"
    )
    path.write_text(body, encoding="utf-8")
    if context_sidecar is not None:
        side = prompts_dir / f"{base}.context.json"
        side.write_text(
            json.dumps(context_sidecar, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    return path


class PromptExporter:
    """Per-run prompt export with monotonic sequence (async-safe)."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir.resolve()
        self._lock = asyncio.Lock()
        self._seq = _initial_sequence_from_disk(self.run_dir)
        self._rows: list[dict[str, Any]] = []

    async def next_sequence(self) -> int:
        async with self._lock:
            self._seq += 1
            return self._seq

    async def record(
        self,
        *,
        provider: str,
        model: str,
        system: str,
        user: str,
        config: Mapping[str, Any],
        context_sidecar: dict[str, Any] | None = None,
    ) -> Path | None:
        if not prompts_export_enabled():
            return None
        meta = current_prompt_call_meta()
        node = meta.node if meta is not None else "unknown"
        iteration = meta.iteration if meta is not None else None
        side = context_sidecar
        if side is None and meta is not None and meta.analyst_render_context is not None and node == "fan_out":
            side = meta.analyst_render_context
        seq = await self.next_sequence()
        rel = export_prompt(
            self.run_dir,
            node=node,
            iteration=iteration,
            provider=provider,
            model=model,
            system=system,
            user=user,
            config=dict(config),
            sequence=seq,
            context_sidecar=side,
        )
        ts = datetime.now(tz=UTC).replace(microsecond=0).isoformat()
        try:
            rel_path = str(rel.relative_to(self.run_dir))
        except ValueError:
            rel_path = str(rel)
        self._rows.append(
            {
                "sequence": seq,
                "timestamp": ts,
                "node": node,
                "iteration": iteration,
                "provider": provider,
                "model": model,
                "system_chars": len(system),
                "user_chars": len(user),
                "file_path": rel_path,
            }
        )
        return rel

    def finalize_index(self) -> Path | None:
        if not prompts_export_enabled() or not self._rows:
            return None
        prompts_dir = self.run_dir / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Prompt export index",
            "",
            "| sequence | timestamp | node | iteration | provider | model | system_chars | user_chars | file_path |",
            "| ---: | --- | --- | --- | --- | --- | ---: | ---: | --- |",
        ]
        for r in sorted(self._rows, key=lambda x: int(x["sequence"])):
            it = r["iteration"]
            it_cell = "" if it is None else str(it)
            lines.append(
                "| {seq:04d} | {ts} | {node} | {it} | {prov} | {model} | {sc} | {uc} | `{fp}` |".format(
                    seq=int(r["sequence"]),
                    ts=r["timestamp"],
                    node=r["node"],
                    it=it_cell,
                    prov=r["provider"],
                    model=r["model"],
                    sc=int(r["system_chars"]),
                    uc=int(r["user_chars"]),
                    fp=r["file_path"],
                )
            )
        idx = prompts_dir / "prompts_index.md"
        idx.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return idx


def _initial_sequence_from_disk(run_dir: Path) -> int:
    d = run_dir / "prompts"
    if not d.is_dir():
        return 0
    best = 0
    for p in d.glob("*.md"):
        if p.name == "prompts_index.md":
            continue
        m = re.match(r"^(\d{4})_", p.name)
        if m:
            try:
                best = max(best, int(m.group(1)))
            except ValueError:
                continue
    return best


async def maybe_export_prompt(
    *,
    provider: str,
    model: str,
    system: str,
    user: str,
    config: Mapping[str, Any],
    context_sidecar: dict[str, Any] | None = None,
) -> None:
    exp = current_prompt_exporter()
    if exp is None:
        return
    split = _logical_split_ctx.get()
    sys_t, usr_t = (split[0], split[1]) if split is not None else (system, user)
    await exp.record(
        provider=provider,
        model=model,
        system=sys_t,
        user=usr_t,
        config=config,
        context_sidecar=context_sidecar,
    )
