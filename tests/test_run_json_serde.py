from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum

from equity_analyst.run_json_serde import (
    canonical_run_document_dict,
    format_run_json_for_disk,
    json_default_for_run_json,
)


class _Tag(Enum):
    X = "x"


def test_canonical_round_trip_datetime_and_decimal() -> None:
    dt = datetime(2026, 5, 14, 12, 30, tzinfo=UTC)
    src = {"ts": dt, "n": Decimal("12.34"), "plain": 7}
    doc = canonical_run_document_dict(src)
    assert isinstance(doc["ts"], str)
    assert "2026-05-14" in doc["ts"]
    assert doc["n"] == "12.34"
    assert doc["plain"] == 7
    again = canonical_run_document_dict(doc)
    assert again == doc


def test_format_run_json_for_disk_is_valid_json() -> None:
    body = format_run_json_for_disk({"a": 1, "b": Decimal("2")})
    assert body.endswith("\n")
    parsed = json.loads(body)
    assert parsed["a"] == 1
    assert parsed["b"] == "2"


def test_json_default_enum_and_fallback() -> None:
    assert json_default_for_run_json(_Tag.X) == "x"
    assert isinstance(json_default_for_run_json(object()), str)


def test_canonical_round_trip_enum_nested() -> None:
    doc = canonical_run_document_dict({"e": _Tag.X, "n": {"inner": Decimal("1.5")}})
    assert doc["e"] == "x"
    assert doc["n"]["inner"] == "1.5"
    assert canonical_run_document_dict(doc) == doc
