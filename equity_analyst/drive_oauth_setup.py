"""Bootstrap OAuth user credentials for Google Drive uploads (personal Gmail / My Drive)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from equity_analyst.config import (
    load_config,
    resolve_drive_oauth_client_secrets_path,
    resolve_drive_oauth_client_secrets_path_from_optional,
    resolve_drive_oauth_token_path,
    resolve_drive_oauth_token_path_from_optional,
)

_OAUTH_DRIVE_FILE_SCOPE = "https://www.googleapis.com/auth/drive.file"


def _print_missing_client_secrets_help(*, expected_path: Path) -> None:
    print(
        f"OAuth client secrets file not found at:\n  {expected_path}\n\n"
        "Create a Desktop OAuth client in Google Cloud Console:\n"
        "  1. Open Google Cloud Console → APIs & Services → Credentials\n"
        "  2. Create credentials → OAuth client ID\n"
        "  3. Application type: Desktop app\n"
        "  4. Download the JSON and save it to the path above\n\n"
        "Set the path explicitly with:\n"
        "  • YAML key drive_oauth_client_secrets_path, or\n"
        "  • Environment variable DRIVE_OAUTH_CLIENT_SECRETS_PATH\n",
        file=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    load_dotenv(override=False)
    parser = argparse.ArgumentParser(
        description="Authorize Google Drive (OAuth user) and save a refresh token for equity_analyst uploads.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config path (uses drive_oauth_client_secrets_path / drive_oauth_token_path when set).",
    )
    args = parser.parse_args(argv)

    if args.config:
        cfg = load_config(args.config)
        client_path = resolve_drive_oauth_client_secrets_path(cfg)
        token_path = resolve_drive_oauth_token_path(cfg)
    else:
        client_path = resolve_drive_oauth_client_secrets_path_from_optional(
            os.environ.get("DRIVE_OAUTH_CLIENT_SECRETS_PATH"),
        )
        token_path = resolve_drive_oauth_token_path_from_optional(os.environ.get("DRIVE_OAUTH_TOKEN_PATH"))

    if not client_path.is_file():
        _print_missing_client_secrets_help(expected_path=client_path)
        return 2

    from google_auth_oauthlib.flow import InstalledAppFlow  # type: ignore[import-untyped]

    flow = InstalledAppFlow.from_client_secrets_file(
        str(client_path),
        scopes=[_OAUTH_DRIVE_FILE_SCOPE],
    )
    creds = flow.run_local_server(port=0)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(creds.to_json(), encoding="utf-8")
    print(f"OAuth token saved to {token_path}. Drive upload ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
