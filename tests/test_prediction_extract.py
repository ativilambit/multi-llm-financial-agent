from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from equity_analyst.config import RunConfig
from equity_analyst.prediction_extract import (
    PREDICTION_HORIZONS,
    extract_predictions_from_synthesis,
    parse_prediction_extract_json,
    rows_from_parsed_payload,
    run_prediction_extract_for_run_dir,
)
from equity_analyst.types import ProviderResponse, ProviderUsage


def _minimal_run_config() -> RunConfig:
    return RunConfig.model_validate(
        {
            "symbol": "ZZ",
            "today_date": "Mon May 11 2026",
            "today_session": "regular",
            "earnings_date": "Mon May 11 2026",
            "next_trading_day": "Tue May 12 2026",
            "followup_open_date": "Tue May 12 2026",
            "providers": ["openai"],
            "prediction_extract_provider": "gemini",
            "prediction_extract_model": "gemini-3-flash-preview",
            "prediction_extract_max_output_tokens": 2048,
            "prediction_extract_timeout_s": 120,
        }
    )


@pytest.mark.asyncio
async def test_extract_predictions_valid_json_builds_five_horizons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "horizons": {
            "earnings_day_open": {
                "probability_up": 0.62,
                "range_low": 10.5,
                "range_high": 11.2,
                "point": None,
            },
            "earnings_day_close": {
                "probability_up": None,
                "range_low": None,
                "range_high": None,
                "point": 10.9,
            },
            "next_trading_day_open": {
                "probability_up": 0.55,
                "range_low": None,
                "range_high": None,
                "point": None,
            },
            "next_trading_day_close": {
                "probability_up": None,
                "range_low": 9.8,
                "range_high": 11.0,
                "point": None,
            },
            "one_week_later_close": {
                "probability_up": 0.48,
                "range_low": None,
                "range_high": None,
                "point": None,
            },
        },
        "confidence": "high",
        "notes": "none",
    }

    async def _fake_invoke(**_kw: object) -> ProviderResponse:
        return ProviderResponse(
            provider_name="gemini",
            model="gemini-3-flash-preview",
            text=json.dumps(payload),
            usage=ProviderUsage(),
            latency_s=0.1,
            raw=None,
        )

    monkeypatch.setattr(
        "equity_analyst.prediction_extract._invoke_prediction_extract_llm",
        _fake_invoke,
    )

    cfg = _minimal_run_config()
    rows = await extract_predictions_from_synthesis(
        synthesis_text="# Synth\nSome text",
        symbol=cfg.symbol,
        run_id="ZZ_20260511T000000Z",
        config=cfg,
    )
    assert len(rows) == 5
    by_h = {r.horizon: r for r in rows}
    assert by_h["earnings_day_open"].predicted_probability_up == pytest.approx(0.62)
    assert by_h["earnings_day_open"].predicted_range_low == pytest.approx(10.5)
    assert by_h["earnings_day_open"].predicted_range_high == pytest.approx(11.2)
    assert by_h["earnings_day_close"].predicted_point == pytest.approx(10.9)
    assert by_h["next_trading_day_close"].predicted_range_low == pytest.approx(9.8)
    assert all(r.source == "llm_extracted" for r in rows)


@pytest.mark.asyncio
async def test_extract_malformed_json_returns_empty_and_warns(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def _fake_invoke(**_kw: object) -> ProviderResponse:
        return ProviderResponse(
            provider_name="gemini",
            model="gemini-3-flash-preview",
            text="not json at all {{{",
            usage=ProviderUsage(),
            latency_s=0.1,
            raw=None,
        )

    monkeypatch.setattr(
        "equity_analyst.prediction_extract._invoke_prediction_extract_llm",
        _fake_invoke,
    )
    caplog.set_level(logging.WARNING)
    cfg = _minimal_run_config()
    rows = await extract_predictions_from_synthesis(
        synthesis_text="body",
        symbol="Z",
        run_id="Z_1",
        config=cfg,
    )
    assert rows == []
    assert any("prediction_extract JSON parse failed" in r.message for r in caplog.records)


def test_probability_down_converts_to_probability_up() -> None:
    data = parse_prediction_extract_json(
        json.dumps(
            {
                "horizons": {
                    "earnings_day_open": {"probability_down": 0.4},
                    "earnings_day_close": {},
                    "next_trading_day_open": {},
                    "next_trading_day_close": {},
                    "one_week_later_close": {},
                },
                "confidence": "low",
                "notes": "",
            }
        )
    )
    assert data is not None
    rows = rows_from_parsed_payload(run_id="R1", data=data)
    o = next(r for r in rows if r.horizon == "earnings_day_open")
    assert o.predicted_probability_up == pytest.approx(0.6)


@pytest.mark.asyncio
async def test_replace_predictions_second_call_deletes_then_inserts_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from equity_analyst.db_ops import best_effort_replace_predictions

    exec_kinds: list[str] = []

    class FakeSession:
        async def execute(self, *args: object) -> None:
            stmt = args[0]
            s = str(stmt).lower()
            if "delete" in s and "prediction" in s:
                exec_kinds.append("delete")
            elif "insert" in s and "prediction" in s:
                exec_kinds.append("insert")
            else:
                exec_kinds.append("other")

        async def commit(self) -> None:
            pass

        async def __aenter__(self) -> FakeSession:
            return self

        async def __aexit__(self, *_a: object) -> None:
            return None

    @contextlib.asynccontextmanager
    async def _fake_session(**_kw: object):
        yield FakeSession()

    monkeypatch.setattr("equity_analyst.db_ops.is_db_available", AsyncMock(return_value=True))
    monkeypatch.setattr("equity_analyst.db_ops.get_async_session", _fake_session)

    rows = [
        {
            "run_id": "ABC_20260101T000000Z",
            "horizon": "earnings_day_open",
            "predicted_probability_up": 0.5,
            "predicted_range_low": None,
            "predicted_range_high": None,
            "predicted_point": None,
            "source": "llm_extracted",
        }
    ]
    assert await best_effort_replace_predictions(
        cfg_db_enabled=True, run_id=rows[0]["run_id"], rows=rows, run_profile="production"
    )
    assert await best_effort_replace_predictions(
        cfg_db_enabled=True, run_id=rows[0]["run_id"], rows=rows, run_profile="production"
    )
    assert exec_kinds == ["delete", "insert", "delete", "insert"]


@pytest.mark.asyncio
async def test_run_prediction_extract_invokes_replace_twice_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    run_dir = Path("outputs") / "ZZ_20260511T000000Z"
    run_dir.mkdir(parents=True)
    cfg = _minimal_run_config()
    (run_dir / "run.json").write_text(
        json.dumps(
            {"run_profile": "production", "config": cfg.model_dump()},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "synthesis.md").write_text("# Hello\n", encoding="utf-8")

    payload = {
        "horizons": {h: {"probability_up": 0.1} for h in PREDICTION_HORIZONS},
        "confidence": "medium",
        "notes": "n",
    }

    async def _fake_invoke(**_kw: object) -> ProviderResponse:
        return ProviderResponse(
            provider_name="gemini",
            model="gemini-3-flash-preview",
            text=json.dumps(payload),
            usage=ProviderUsage(),
            latency_s=0.01,
            raw=None,
        )

    monkeypatch.setattr(
        "equity_analyst.prediction_extract._invoke_prediction_extract_llm",
        _fake_invoke,
    )

    replace = AsyncMock(return_value=True)
    monkeypatch.setattr("equity_analyst.prediction_extract.db_replace_predictions", replace)

    await run_prediction_extract_for_run_dir(run_dir=run_dir, cfg=cfg)
    await run_prediction_extract_for_run_dir(run_dir=run_dir, cfg=cfg)
    assert replace.await_count == 2
    first_kw = replace.await_args_list[0].kwargs
    assert first_kw["run_id"] == run_dir.name
    assert len(first_kw["rows"]) == 5
    assert first_kw["env"] == "production"


@pytest.mark.asyncio
async def test_run_prediction_extract_uses_db_synthesis_when_file_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    run_dir = Path("outputs") / "ZZ_20260511T000000Z"
    run_dir.mkdir(parents=True)
    cfg = _minimal_run_config()
    (run_dir / "run.json").write_text(
        json.dumps(
            {"run_profile": "production", "config": cfg.model_dump()},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    payload = {
        "horizons": {h: {"probability_up": 0.2} for h in PREDICTION_HORIZONS},
        "confidence": "low",
        "notes": "",
    }

    calls: list[dict[str, str]] = []

    async def _fake_invoke(*, user_message: str, **_kw: object) -> ProviderResponse:
        calls.append({"user_message": user_message})
        return ProviderResponse(
            provider_name="gemini",
            model="gemini-3-flash-preview",
            text=json.dumps(payload),
            usage=ProviderUsage(),
            latency_s=0.01,
            raw=None,
        )

    monkeypatch.setattr(
        "equity_analyst.prediction_extract._invoke_prediction_extract_llm",
        _fake_invoke,
    )
    monkeypatch.setattr(
        "equity_analyst.prediction_extract.load_synthesis_markdown_from_db",
        AsyncMock(return_value="# Synth from DB\n"),
    )
    replace = AsyncMock(return_value=True)
    monkeypatch.setattr("equity_analyst.prediction_extract.db_replace_predictions", replace)

    await run_prediction_extract_for_run_dir(run_dir=run_dir, cfg=cfg)
    assert replace.await_count == 1
    assert len(calls) == 1
    assert "Synth from DB" in calls[0]["user_message"]


def test_parse_strips_markdown_fences() -> None:
    inner = {
        "horizons": {h: {} for h in PREDICTION_HORIZONS},
        "confidence": "high",
        "notes": "",
    }
    text = "```json\n" + json.dumps(inner) + "\n```"
    assert parse_prediction_extract_json(text) is not None
