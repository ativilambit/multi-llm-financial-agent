from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from equity_analyst.gemini_cache import CacheEntry, GeminiCacheIndex, prefix_sha256


def test_prefix_sha256_stable() -> None:
    assert prefix_sha256("hello") == prefix_sha256("hello")
    assert prefix_sha256("hello") != prefix_sha256("hello ")


def test_index_store_lookup(tmp_path: Path) -> None:
    idx = GeminiCacheIndex(path=tmp_path / "idx.json")
    assert idx.lookup("prefix-a", "gemini-2.5-flash") is None
    idx.store("prefix-a", "gemini-2.5-flash", "cachedContents/abc", 3600)
    assert idx.lookup("prefix-a", "gemini-2.5-flash") == "cachedContents/abc"
    raw = json.loads((tmp_path / "idx.json").read_text(encoding="utf-8"))
    assert "prefix-a" not in json.dumps(raw)
    key = f"{prefix_sha256('prefix-a')}:gemini-2.5-flash"
    assert key in raw["entries"]


def test_index_model_isolation(tmp_path: Path) -> None:
    idx = GeminiCacheIndex(path=tmp_path / "idx.json")
    idx.store("p", "gemini-2.5-flash", "cachedContents/f1", 3600)
    idx.store("p", "gemini-2.5-pro", "cachedContents/f2", 3600)
    assert idx.lookup("p", "gemini-2.5-flash") == "cachedContents/f1"
    assert idx.lookup("p", "gemini-2.5-pro") == "cachedContents/f2"


def test_index_expiry_removes_entry(tmp_path: Path) -> None:
    idx = GeminiCacheIndex(path=tmp_path / "idx.json")
    past = (datetime.now(tz=UTC) - timedelta(hours=1)).isoformat()
    key = f"{prefix_sha256('old')}:gemini-2.5-flash"
    ent = CacheEntry(
        cache_name="cachedContents/stale",
        model="gemini-2.5-flash",
        prefix_hash=prefix_sha256("old"),
        created_at=past,
        ttl_s=3600,
        expires_at=past,
    )
    data = {"entries": {key: ent.__dict__}}
    (tmp_path / "idx.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    assert idx.lookup("old", "gemini-2.5-flash") is None
    raw = json.loads((tmp_path / "idx.json").read_text(encoding="utf-8"))
    assert raw["entries"] == {}


def test_cleanup_drops_expired(tmp_path: Path) -> None:
    idx = GeminiCacheIndex(path=tmp_path / "idx.json")
    now = datetime.now(tz=UTC)
    past = (now - timedelta(seconds=10)).isoformat()
    future = (now + timedelta(hours=1)).isoformat()
    key_old = f"{prefix_sha256('a')}:m"
    key_new = f"{prefix_sha256('b')}:m"
    data = {
        "entries": {
            key_old: {
                "cache_name": "cachedContents/o",
                "model": "m",
                "prefix_hash": prefix_sha256("a"),
                "created_at": past,
                "ttl_s": 1,
                "expires_at": past,
            },
            key_new: {
                "cache_name": "cachedContents/n",
                "model": "m",
                "prefix_hash": prefix_sha256("b"),
                "created_at": now.isoformat(),
                "ttl_s": 3600,
                "expires_at": future,
            },
        }
    }
    (tmp_path / "idx.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    idx.cleanup()
    raw = json.loads((tmp_path / "idx.json").read_text(encoding="utf-8"))
    assert key_old not in raw["entries"]
    assert key_new in raw["entries"]
