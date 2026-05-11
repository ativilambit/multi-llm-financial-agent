from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from equity_analyst.iterative import (
    _CHECKPOINT_BASENAMES,
    _delete_checkpoint_files,
    maybe_delete_iterative_checkpoint,
)


def test_delete_checkpoint_files_removes_sqlite_siblings(tmp_path: Path) -> None:
    log = logging.getLogger("test_ckpt")
    for name in _CHECKPOINT_BASENAMES:
        (tmp_path / name).write_text("x", encoding="utf-8")
    _delete_checkpoint_files(tmp_path, log)
    for name in _CHECKPOINT_BASENAMES:
        assert not (tmp_path / name).is_file()


def test_maybe_delete_iterative_checkpoint_false_keeps_files(tmp_path: Path) -> None:
    (tmp_path / "checkpoint.sqlite").write_text("db", encoding="utf-8")
    with patch("equity_analyst.iterative._delete_checkpoint_files") as mock_del:
        maybe_delete_iterative_checkpoint(
            tmp_path,
            delete_checkpoint_after_success=False,
        )
        mock_del.assert_not_called()
    assert (tmp_path / "checkpoint.sqlite").is_file()


def test_maybe_delete_iterative_checkpoint_true_removes_files(tmp_path: Path) -> None:
    (tmp_path / "checkpoint.sqlite").write_text("db", encoding="utf-8")
    maybe_delete_iterative_checkpoint(tmp_path, delete_checkpoint_after_success=True)
    assert not (tmp_path / "checkpoint.sqlite").is_file()


def test_delete_checkpoint_files_warns_on_error(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    log = logging.getLogger("test_ckpt_warn")
    p = tmp_path / "checkpoint.sqlite"
    p.write_text("db", encoding="utf-8")

    def boom(*_a: object, **_k: object) -> None:
        raise OSError("simulated")

    with (
        patch.object(Path, "is_file", return_value=True),
        patch.object(Path, "unlink", boom),
        caplog.at_level(logging.WARNING, logger="test_ckpt_warn"),
    ):
        _delete_checkpoint_files(tmp_path, log)
    assert any("Failed to remove checkpoint artifact" in r.message for r in caplog.records)
