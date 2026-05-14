"""Canonical serialization for ``run.json`` on disk and ``runs.run_document`` (JSONB).

Use the same formatting everywhere so the persisted dict matches the file bytes
(modulo the trailing newline) and drift-free upserts are possible.
"""

from __future__ import annotations

import json
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from typing import Any, cast


def json_default_for_run_json(obj: Any) -> Any:
    """JSON ``default`` hook for values that appear in run metadata dicts."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, time):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, Enum):
        return obj.value
    return str(obj)


def format_run_json_for_disk(meta: dict[str, Any]) -> str:
    """Return ``run.json`` file body (including trailing newline)."""
    return json.dumps(meta, indent=2, sort_keys=True, default=json_default_for_run_json) + "\n"


def canonical_run_document_dict(meta: dict[str, Any]) -> dict[str, Any]:
    """Parse-serialize round-trip using the same rules as :func:`format_run_json_for_disk`."""
    blob = json.loads(format_run_json_for_disk(meta))
    if not isinstance(blob, dict):
        raise TypeError("canonical_run_document_dict expected a JSON object")
    return cast(dict[str, Any], blob)
