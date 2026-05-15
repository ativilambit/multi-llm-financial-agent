from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from equity_analyst.config import (
    RunConfig,
    env_from_persisted_run_json,
    run_profile_from_persisted_run_json,
)
from equity_analyst.db_ops import (
    best_effort_replace_predictions as db_replace_predictions,
)
from equity_analyst.db_ops import (
    load_run_document_from_db,
    load_synthesis_markdown_from_db,
)
from equity_analyst.outcome_tracker import _pick_synthesis_path
from equity_analyst.prompt_export import logical_prompt_split, prompt_call_context
from equity_analyst.prompt_parts import _load_prompt_file
from equity_analyst.providers.registry import ProviderRegistry
from equity_analyst.types import ProviderResponse

logger = logging.getLogger(__name__)

PREDICTION_HORIZONS: tuple[str, ...] = (
    "earnings_day_open",
    "earnings_day_close",
    "next_trading_day_open",
    "next_trading_day_close",
    "one_week_later_close",
)

_SOURCE_LLM = "llm_extracted"
_MAX_SYNTHESIS_CHARS = 240_000


@dataclass(frozen=True)
class PredictionRow:
    """Structured values for one ``predictions`` table row (pre-insert)."""

    run_id: str
    horizon: str
    predicted_probability_up: float | None
    predicted_range_low: float | None
    predicted_range_high: float | None
    predicted_point: float | None
    source: str


def _strip_markdown_fences(t: str) -> str:
    t = t.strip()
    if not t.startswith("```"):
        return t
    lines = t.split("\n")
    body = "\n".join(lines[1:]) if lines else ""
    if "```" in body:
        body = body.rsplit("```", 1)[0]
    return body.strip()


def _balanced_brace_objects(s: str) -> list[str]:
    n = len(s)
    out: list[str] = []
    i = 0
    while i < n:
        if s[i] != "{":
            i += 1
            continue
        depth = 0
        in_str = False
        esc = False
        start = i
        j = i
        found = False
        while j < n:
            c = s[j]
            if esc:
                esc = False
                j += 1
                continue
            if in_str:
                if c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                j += 1
                continue
            if c == '"':
                in_str = True
                j += 1
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    out.append(s[start : j + 1])
                    found = True
                    break
            j += 1
        i = j + 1 if found else i + 1
    return out


def _rstrip_trailing_json_commas(s: str) -> str:
    t = s.rstrip()
    while t.endswith(","):
        t = t[:-1].rstrip()
    return t


def _prediction_payload_has_horizons(data: dict[str, Any]) -> bool:
    h = data.get("horizons")
    if not isinstance(h, dict):
        return False
    return any(k in h for k in PREDICTION_HORIZONS)


def _candidate_dicts_from_text(text: str) -> list[tuple[dict[str, Any], int]]:
    seen: set[str] = set()
    out: list[tuple[dict[str, Any], int]] = []

    def _push(raw_slice: str, d: dict[str, Any]) -> None:
        key = json.dumps(d, sort_keys=True)
        if key in seen:
            return
        seen.add(key)
        out.append((d, len(raw_slice)))

    stripped = _strip_markdown_fences(text.strip())
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            _push(stripped, obj)
    except json.JSONDecodeError:
        pass
    for sub in _balanced_brace_objects(text):
        try:
            obj = json.loads(sub)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            _push(sub, obj)
    return out


_REPAIR_SUFFIXES: tuple[str, ...] = (
    '}},"confidence":"low","notes":"truncated"}',
    '}},"confidence":"low","notes":""}',
    "}}}",
    "}",
)


def _attempt_truncated_prediction_json_repair(text: str) -> dict[str, Any] | None:
    t = _strip_markdown_fences(text.strip())
    start = t.find("{")
    if start < 0:
        return None
    base_full = t[start:].rstrip()
    if not base_full:
        return None
    tried: set[str] = set()

    def _try_candidate(candidate: str) -> dict[str, Any] | None:
        if candidate in tried:
            return None
        tried.add(candidate)
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        if isinstance(obj, dict) and _prediction_payload_has_horizons(obj):
            return obj
        return None

    max_trim = min(512, max(0, len(base_full) - 1))
    for trim in range(0, max_trim + 1):
        base = base_full[:-trim].rstrip() if trim else base_full
        if not base:
            continue
        for b in {base, _rstrip_trailing_json_commas(base)}:
            if not b:
                continue
            for suf in _REPAIR_SUFFIXES:
                got = _try_candidate(b + suf)
                if got is not None:
                    return got
    return None


def parse_prediction_extract_json(text: str) -> dict[str, Any] | None:
    """Parse model output into a dict with ``horizons`` / ``confidence`` / ``notes`` or ``None``."""
    candidates = _candidate_dicts_from_text(text)
    best: dict[str, Any] | None = None
    best_rank: tuple[int, int] = (-1, -1)
    for data, raw_len in candidates:
        if not _prediction_payload_has_horizons(data):
            continue
        h = data["horizons"]
        assert isinstance(h, dict)
        filled = sum(
            1
            for k in PREDICTION_HORIZONS
            if isinstance(h.get(k), dict) and any(v is not None for v in (h.get(k) or {}).values())
        )
        rank = (filled, raw_len)
        if rank > best_rank:
            best_rank = rank
            best = data
    if best is not None:
        return best
    repaired = _attempt_truncated_prediction_json_repair(text)
    if repaired is not None:
        logger.warning("prediction_extract: salvaged truncated JSON for horizons")
        return repaired
    logger.warning(
        "prediction_extract JSON parse failed (no horizons object); model_raw=%s",
        text[:1200] if text else "",
    )
    return None


def _coerce_float(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip().replace("$", "").replace(",", "")
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _coerce_probability_up(blob: dict[str, Any]) -> float | None:
    up = _coerce_float(blob.get("probability_up"))
    if up is not None:
        return max(0.0, min(1.0, up))
    down = _coerce_float(blob.get("probability_down"))
    if down is not None:
        return max(0.0, min(1.0, 1.0 - down))
    return None


def rows_from_parsed_payload(*, run_id: str, data: dict[str, Any]) -> list[PredictionRow]:
    raw_h = data.get("horizons")
    if not isinstance(raw_h, dict):
        return []
    rows: list[PredictionRow] = []
    for hz in PREDICTION_HORIZONS:
        blob = raw_h.get(hz)
        if not isinstance(blob, dict):
            blob = {}
        p_up = _coerce_probability_up(blob)
        r_lo = _coerce_float(blob.get("range_low"))
        r_hi = _coerce_float(blob.get("range_high"))
        pt = _coerce_float(blob.get("point"))
        rows.append(
            PredictionRow(
                run_id=run_id,
                horizon=hz,
                predicted_probability_up=p_up,
                predicted_range_low=r_lo,
                predicted_range_high=r_hi,
                predicted_point=pt,
                source=_SOURCE_LLM,
            )
        )
    return rows


def _user_message(*, symbol: str, run_id: str, synthesis_text: str) -> str:
    body = (
        synthesis_text
        if len(synthesis_text) <= _MAX_SYNTHESIS_CHARS
        else (
            synthesis_text[:_MAX_SYNTHESIS_CHARS]
            + "\n\n...(synthesis truncated for extraction request; prefer signals from the excerpt above)..."
        )
    )
    return f"Symbol: {symbol}\nRun ID: {run_id}\n\n## Synthesis markdown\n\n{body}\n"


async def _invoke_prediction_extract_llm(
    *,
    user_message: str,
    config: RunConfig,
) -> ProviderResponse:
    reg = ProviderRegistry.default()
    provider = reg.create(
        config.prediction_extract_provider,
        model=config.prediction_extract_model,
        gemini_cache_index=None,
    )
    system = _load_prompt_file("prediction_extract_system.md")
    prompt = f"{system}\n\n---\n\n{user_message}"
    timeout_s = float(config.prediction_extract_timeout_s)

    async def _call() -> ProviderResponse:
        with (
            prompt_call_context(node="prediction_extract"),
            logical_prompt_split(system, user_message),
        ):
            return await provider.generate(
                prompt,
                enable_web_search=False,
                max_output_tokens=int(config.prediction_extract_max_output_tokens),
            )

    return await asyncio.wait_for(_call(), timeout=timeout_s)


async def extract_predictions_from_synthesis(
    *,
    synthesis_text: str,
    symbol: str,
    run_id: str,
    config: RunConfig,
) -> list[PredictionRow]:
    """Call the configured extractor LLM and map JSON into five horizon rows (or [])."""
    if not synthesis_text.strip():
        logger.warning("prediction_extract: empty synthesis run_id=%s", run_id)
        return []
    try:
        resp = await _invoke_prediction_extract_llm(
            user_message=_user_message(symbol=symbol, run_id=run_id, synthesis_text=synthesis_text),
            config=config,
        )
    except TimeoutError:
        logger.warning(
            "prediction_extract: LLM timeout run_id=%s timeout_s=%s",
            run_id,
            config.prediction_extract_timeout_s,
        )
        return []
    except Exception as exc:
        logger.warning("prediction_extract: LLM call failed run_id=%s error=%r", run_id, exc)
        return []

    parsed = parse_prediction_extract_json(resp.text)
    if parsed is None:
        return []
    return rows_from_parsed_payload(run_id=run_id, data=parsed)


def _write_predictions_fallback_json(
    *, run_dir: Path, run_id: str, symbol: str, rows: list[PredictionRow]
) -> None:
    path = run_dir / "predictions_extract.json"
    payload = {
        "run_id": run_id,
        "symbol": symbol,
        "written_at_utc": datetime.now(tz=UTC).replace(microsecond=0).isoformat(),
        "source": _SOURCE_LLM,
        "rows": [asdict(r) for r in rows],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("prediction_extract: wrote fallback artifact path=%s", path)


async def run_prediction_extract_for_run_dir(
    *, run_dir: Path, cfg: RunConfig
) -> list[PredictionRow]:
    """Read final synthesis for ``run_dir``, extract rows, persist to DB or fallback JSON."""
    try:
        run_dir = run_dir.expanduser().resolve()
        run_id = run_dir.name
        syn_path = _pick_synthesis_path(run_dir)
        synthesis_text = ""
        if syn_path.is_file():
            synthesis_text = syn_path.read_text(encoding="utf-8")
        else:
            db_syn = await load_synthesis_markdown_from_db(run_id, database_url=cfg.database_url)
            if db_syn:
                synthesis_text = db_syn
            else:
                logger.warning(
                    "prediction_extract: missing synthesis file path=%s and no runs.synthesis_markdown",
                    syn_path,
                )
                return []
        symbol = cfg.symbol
        rows = await extract_predictions_from_synthesis(
            synthesis_text=synthesis_text,
            symbol=symbol,
            run_id=run_id,
            config=cfg,
        )
        dict_rows = [
            {
                "run_id": r.run_id,
                "horizon": r.horizon,
                "predicted_probability_up": r.predicted_probability_up,
                "predicted_range_low": r.predicted_range_low,
                "predicted_range_high": r.predicted_range_high,
                "predicted_point": r.predicted_point,
                "source": r.source,
            }
            for r in rows
        ]
        run_json_path = run_dir / "run.json"
        db_profile = cfg.run_profile
        db_env = cfg.env
        blob: dict[str, Any] | None = None
        if run_json_path.is_file():
            try:
                raw = json.loads(run_json_path.read_text(encoding="utf-8"))
                blob = raw if isinstance(raw, dict) else None
            except Exception:
                blob = None
        if blob is None:
            blob = await load_run_document_from_db(run_id, database_url=cfg.database_url)
        if isinstance(blob, dict):
            try:
                db_profile = run_profile_from_persisted_run_json(blob)
                db_env = env_from_persisted_run_json(blob)
            except Exception:
                db_profile = cfg.run_profile
                db_env = cfg.env
        ok = await db_replace_predictions(
            cfg_db_enabled=cfg.db_enabled,
            run_id=run_id,
            rows=dict_rows,
            database_url=cfg.database_url,
            run_profile=db_profile,
            env=db_env,
        )
        if not ok and rows:
            _write_predictions_fallback_json(
                run_dir=run_dir, run_id=run_id, symbol=symbol, rows=rows
            )
        elif not ok and not rows:
            logger.warning("prediction_extract: no rows and DB not updated run_id=%s", run_id)
        return rows
    except Exception as exc:
        logger.warning(
            "prediction_extract: unexpected failure run_dir=%s error=%r",
            str(run_dir),
            exc,
        )
        return []
