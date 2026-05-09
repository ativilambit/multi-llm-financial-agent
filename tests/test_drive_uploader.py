from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from googleapiclient.errors import HttpError

from equity_analyst.drive_uploader import (
    DriveUploader,
    maybe_upload_run_to_drive_raw,
)


def _http_error(status: int) -> HttpError:
    resp = MagicMock()
    resp.status = status
    return HttpError(resp, b"")


@pytest.fixture
def sa_json(tmp_path: Path) -> Path:
    p = tmp_path / "sa.json"
    p.write_text(
        json.dumps(
            {
                "type": "service_account",
                "project_id": "test",
                "private_key_id": "x",
                "private_key": "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA0Z3VS5JJcds3xfn/ygWyF8PbnGy0AHB7MhgwKVSmUB4BF1Er\n"
                "Q8WP2mNf1F7YbZRbQ0BKzavSvLAul6BnNQN0OJmjJMyExSjX4mlfYU7cJ8RyzHq\n"
                "QIDAQAB\n-----END RSA PRIVATE KEY-----\n",
                "client_email": "sa@test.iam.gserviceaccount.com",
                "client_id": "1",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        ),
        encoding="utf-8",
    )
    return p


def test_upload_directory_walks_and_creates_nested_folders(
    tmp_path: Path, sa_json: Path, monkeypatch: Any
) -> None:
    out = tmp_path / "RUN1"
    out.mkdir()
    (out / "top.md").write_text("a", encoding="utf-8")
    it = out / "iterations"
    it.mkdir()
    (it / "inner.json").write_text("{}", encoding="utf-8")
    (out / ".secret").write_text("x", encoding="utf-8")

    executes: list[Any] = [
        {"files": []},
        {"id": "RUN_FOLDER"},
        {"id": "f_top"},
        {"files": []},
        {"id": "ITER_FOLDER"},
        {"id": "f_inner"},
    ]

    def pop_execute() -> Any:
        return executes.pop(0)

    files_api = MagicMock()
    files_api.list.return_value.execute.side_effect = pop_execute
    files_api.create.return_value.execute.side_effect = pop_execute

    svc = MagicMock()
    svc.files.return_value = files_api

    uploader = DriveUploader(sa_json, "ROOT")
    monkeypatch.setattr(uploader, "_ensure_service", lambda: svc)

    url = uploader.upload_directory(out, run_id="RUN1")
    assert url == "https://drive.google.com/drive/folders/RUN_FOLDER"

    list_calls = files_api.list.call_args_list
    create_calls = files_api.create.call_args_list

    assert any("RUN1" in str(c) for c in list_calls)
    assert any("iterations" in str(c) for c in list_calls)
    assert len(create_calls) >= 4
    media_bodies = [c.kwargs.get("media_body") for c in create_calls if c.kwargs.get("media_body")]
    assert len(media_bodies) == 2


@pytest.mark.asyncio
async def test_retry_http_503_then_success(
    tmp_path: Path, sa_json: Path, monkeypatch: Any, caplog: pytest.LogCaptureFixture
) -> None:
    out = tmp_path / "RUN1"
    out.mkdir()
    (out / "one.md").write_text("x", encoding="utf-8")

    files_api = MagicMock()
    files_api.list.return_value.execute.side_effect = [{"files": []}]
    files_api.create.return_value.execute.side_effect = [
        {"id": "RUN_FOLDER"},
        _http_error(503),
        {"id": "file_ok"},
    ]

    svc = MagicMock()
    svc.files.return_value = files_api

    sleeps: list[float] = []

    def fake_sleep(s: float) -> None:
        sleeps.append(s)

    monkeypatch.setattr("equity_analyst.drive_uploader.time.sleep", fake_sleep)

    uploader = DriveUploader(sa_json, "ROOT")
    monkeypatch.setattr(uploader, "_ensure_service", lambda: svc)

    with caplog.at_level(logging.WARNING, logger="equity_analyst.drive_uploader"):
        url = uploader.upload_directory(out, run_id="RUN1")
    assert "RUN_FOLDER" in url
    assert len(sleeps) == 1
    assert sleeps[0] == 1.0
    assert sum(1 for r in caplog.records if "Drive upload retry" in r.message) == 1


@pytest.mark.asyncio
async def test_maybe_upload_missing_credentials_no_crash(tmp_path: Path) -> None:
    out = tmp_path / "o"
    out.mkdir()
    (out / "run.json").write_text("{}", encoding="utf-8")
    url = await maybe_upload_run_to_drive_raw(
        drive_upload_enabled=True,
        drive_credentials_path=str(tmp_path / "nope.json"),
        drive_root_folder_id="ROOT",
        out_dir=out,
        run_id="R",
    )
    assert url is None


def test_skips_dotfiles(tmp_path: Path, sa_json: Path, monkeypatch: Any) -> None:
    out = tmp_path / "RUN1"
    out.mkdir()
    (out / "visible.md").write_text("a", encoding="utf-8")
    (out / ".hidden").write_text("b", encoding="utf-8")

    executes: list[Any] = [
        {"files": []},
        {"id": "RUN_FOLDER"},
        {"id": "only_one_file"},
    ]

    def pop_execute() -> Any:
        return executes.pop(0)

    files_api = MagicMock()
    files_api.list.return_value.execute.side_effect = pop_execute
    files_api.create.return_value.execute.side_effect = pop_execute
    svc = MagicMock()
    svc.files.return_value = files_api

    uploader = DriveUploader(sa_json, "ROOT")
    monkeypatch.setattr(uploader, "_ensure_service", lambda: svc)
    uploader.upload_directory(out, run_id="RUN1")

    create_kw = [c.kwargs for c in files_api.create.call_args_list if "media_body" in c.kwargs]
    assert len(create_kw) == 1


def test_discovery_build_called_once(monkeypatch: Any, sa_json: Path, tmp_path: Path) -> None:
    from google.oauth2 import service_account as sa_mod

    out = tmp_path / "RUN1"
    out.mkdir()
    (out / "a.md").write_text("z", encoding="utf-8")

    built: list[int] = []

    def fake_build(*_a: Any, **_k: Any) -> MagicMock:
        built.append(1)
        files_api = MagicMock()
        files_api.list.return_value.execute.return_value = {"files": [{"id": "existing"}]}
        files_api.create.return_value.execute.side_effect = [{"id": "f1"}, {"id": "f2"}]
        svc = MagicMock()
        svc.files.return_value = files_api
        return svc

    with (
        patch("googleapiclient.discovery.build", side_effect=fake_build),
        patch.object(sa_mod.Credentials, "from_service_account_file", return_value=MagicMock()),
    ):
        u = DriveUploader(sa_json, "ROOT")
        u.upload_directory(out, run_id="RUN1")
        u.upload_directory(out, run_id="RUN1")
    assert len(built) == 1
