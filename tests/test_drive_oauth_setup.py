from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from equity_analyst import drive_oauth_setup
from equity_analyst.drive_uploader import _OAUTH_DRIVE_SCOPES


def test_main_missing_client_secrets_exits_2_with_instructions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    secrets = tmp_path / "client.json"
    monkeypatch.setattr(
        drive_oauth_setup,
        "resolve_drive_oauth_client_secrets_path_from_optional",
        lambda *_a, **_k: secrets,
    )
    monkeypatch.setattr(
        drive_oauth_setup,
        "resolve_drive_oauth_token_path_from_optional",
        lambda *_a, **_k: tmp_path / "token.json",
    )
    rc = drive_oauth_setup.main([])
    err = capsys.readouterr().err
    assert rc == 2
    assert "APIs & Services" in err
    assert "Desktop app" in err
    assert str(secrets) in err


def test_main_success_writes_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    secrets = tmp_path / "client.json"
    secrets.write_text(
        json.dumps(
            {
                "installed": {
                    "client_id": "cid",
                    "client_secret": "sec",
                    "auth_uri": "https://example.com/auth",
                    "token_uri": "https://example.com/token",
                    "redirect_uris": ["http://localhost"],
                }
            }
        ),
        encoding="utf-8",
    )
    token_out = tmp_path / "saved_token.json"
    monkeypatch.setattr(
        drive_oauth_setup,
        "resolve_drive_oauth_client_secrets_path_from_optional",
        lambda *_a, **_k: secrets,
    )
    monkeypatch.setattr(
        drive_oauth_setup,
        "resolve_drive_oauth_token_path_from_optional",
        lambda *_a, **_k: token_out,
    )

    fake_creds = MagicMock()
    fake_creds.to_json.return_value = '{"token": "x"}'

    class FakeFlow:
        @classmethod
        def from_client_secrets_file(cls, *_a: Any, scopes: Any = None, **_k: Any) -> FakeFlow:
            assert scopes == list(_OAUTH_DRIVE_SCOPES)
            return cls()

        def run_local_server(self, *, port: int) -> MagicMock:
            assert port == 0
            return fake_creds

    with patch("google_auth_oauthlib.flow.InstalledAppFlow", FakeFlow):
        rc = drive_oauth_setup.main([])
    assert rc == 0
    assert token_out.is_file()
    assert token_out.read_text(encoding="utf-8") == '{"token": "x"}'
    assert fake_creds.to_json.called


def test_main_with_config_yaml_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "symbol: X",
                "today_date: d",
                "today_session: s",
                "earnings_date: e",
                "earnings_timing: t",
                "target_dates: []",
                "next_trading_day: n",
                "followup_open_date: f",
                "providers: [openai]",
                f"drive_oauth_client_secrets_path: {tmp_path / 'c.json'}",
                f"drive_oauth_token_path: {tmp_path / 't.json'}",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "c.json").write_text(
        json.dumps(
            {
                "installed": {
                    "client_id": "cid",
                    "client_secret": "sec",
                    "auth_uri": "https://example.com/auth",
                    "token_uri": "https://example.com/token",
                    "redirect_uris": ["http://localhost"],
                }
            }
        ),
        encoding="utf-8",
    )
    fake_creds = MagicMock()
    fake_creds.to_json.return_value = "{}"

    class FakeFlow:
        @classmethod
        def from_client_secrets_file(cls, *_a: Any, scopes: Any = None, **_k: Any) -> FakeFlow:
            assert scopes == list(_OAUTH_DRIVE_SCOPES)
            return cls()

        def run_local_server(self, *, port: int) -> MagicMock:
            return fake_creds

    with patch("google_auth_oauthlib.flow.InstalledAppFlow", FakeFlow):
        rc = drive_oauth_setup.main(["--config", str(yaml_path)])
    assert rc == 0
    assert (tmp_path / "t.json").is_file()
