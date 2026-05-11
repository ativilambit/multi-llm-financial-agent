from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_INDEX_PATH = Path("outputs") / ".gemini_cache_index.json"


def prefix_sha256(prefix: str) -> str:
    return hashlib.sha256(prefix.encode("utf-8")).hexdigest()


def gemini_cache_tools_signature(enable_web_search: bool) -> str:
    """Stable cache-index segment; explicit Gemini caches bind tools at creation time."""
    return "google_search" if enable_web_search else "none"


@dataclass
class CacheEntry:
    cache_name: str
    model: str
    prefix_hash: str
    created_at: str
    ttl_s: int
    expires_at: str


class GeminiCacheIndex:
    """Persist explicit Gemini cache names keyed by (sha256(prefix), model, tools_signature)."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or DEFAULT_INDEX_PATH

    def _load(self) -> dict[str, Any]:
        if not self._path.is_file():
            return {"entries": {}}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Gemini cache index unreadable path=%s err=%s", self._path, exc)
            return {"entries": {}}
        if not isinstance(raw, dict):
            return {"entries": {}}
        entries = raw.get("entries")
        if not isinstance(entries, dict):
            raw["entries"] = {}
        return raw

    def _write(self, data: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def lookup(self, prefix: str, model: str, tools_signature: str) -> str | None:
        self.cleanup()
        data = self._load()
        entries: dict[str, Any] = data["entries"]
        key = _entry_key(prefix_sha256(prefix), model, tools_signature)
        ent_raw = entries.get(key)
        if not isinstance(ent_raw, dict):
            return None
        try:
            entry = CacheEntry(
                cache_name=str(ent_raw["cache_name"]),
                model=str(ent_raw["model"]),
                prefix_hash=str(ent_raw["prefix_hash"]),
                created_at=str(ent_raw["created_at"]),
                ttl_s=int(ent_raw["ttl_s"]),
                expires_at=str(ent_raw["expires_at"]),
            )
        except (KeyError, TypeError, ValueError):
            return None
        if entry.prefix_hash != prefix_sha256(prefix) or entry.model != model:
            return None
        try:
            exp = datetime.fromisoformat(entry.expires_at.replace("Z", "+00:00"))
        except ValueError:
            del entries[key]
            self._write(data)
            return None
        if exp <= datetime.now(tz=UTC):
            del entries[key]
            self._write(data)
            return None
        return entry.cache_name

    def store(self, prefix: str, model: str, cache_name: str, ttl_s: int, tools_signature: str) -> None:
        self.cleanup()
        data = self._load()
        entries: dict[str, Any] = data["entries"]
        now = datetime.now(tz=UTC)
        ph = prefix_sha256(prefix)
        key = _entry_key(ph, model, tools_signature)
        expires = now + timedelta(seconds=ttl_s)
        entries[key] = asdict(
            CacheEntry(
                cache_name=cache_name,
                model=model,
                prefix_hash=ph,
                created_at=now.isoformat(),
                ttl_s=ttl_s,
                expires_at=expires.isoformat(),
            )
        )
        self._write(data)

    def cleanup(self) -> None:
        data = self._load()
        entries: dict[str, Any] = data["entries"]
        now = datetime.now(tz=UTC)
        removed = 0
        stale_keys: list[str] = []
        for key, ent_raw in list(entries.items()):
            if not isinstance(ent_raw, dict):
                stale_keys.append(key)
                continue
            exp_s = ent_raw.get("expires_at")
            if not isinstance(exp_s, str):
                stale_keys.append(key)
                continue
            try:
                exp = datetime.fromisoformat(exp_s.replace("Z", "+00:00"))
            except ValueError:
                stale_keys.append(key)
                continue
            if exp <= now:
                stale_keys.append(key)
        for k in stale_keys:
            del entries[k]
            removed += 1
        if removed:
            self._write(data)


def _entry_key(prefix_hash: str, model: str, tools_signature: str) -> str:
    return f"{prefix_hash}:{model}:{tools_signature}"
